"""
comfy_embed.py — Run ComfyUI workflows IN-PROCESS, inside an @spaces.GPU call.

Why this exists
---------------
The Docker/T4 Space runs ComfyUI as a subprocess and the cast pipeline talks
to it over HTTP (the `http` transport in cast_pipeline). That model is
impossible on a ZeroGPU **Gradio** Space: ZeroGPU only grants the GPU inside
`@spaces.GPU`-decorated functions running in the MAIN process. A child
`python main.py` server never enters that allocation path, so it sees no CUDA
device.

So on ZeroGPU we keep ComfyUI's node graph, custom nodes and the exact tuned
workflow dicts the pipeline already builds — but we drive ComfyUI's
`PromptExecutor` directly, in-process, from inside one `@spaces.GPU` call per
image. This mirrors how the LLM already runs (transformers + @spaces.GPU, see
model_client._hf_generate).

Transport contract (matches the HTTP helpers in cast_pipeline so callers don't
change):
    run_workflow(workflow, client_id) -> img_info dict | None
        img_info ~ {"filename", "subfolder", "type", "_path"} (first SaveImage
        output), shaped like cast_pipeline._comfy_wait's return.
    input_dir() / output_dir()  — ComfyUI's real input/output folders, so the
        pipeline's "upload" becomes a file copy and "fetch" a file read.
    health()    — cheap filesystem check (does NOT import torch/comfy).
    stage_models() — download the Anima/SAM/YOLO weights into ComfyUI's model
        dirs at Space startup (Docker bakes these into image layers; a Gradio
        SDK Space has no Dockerfile, so we fetch them here).

Everything here is version-DEFENSIVE on purpose: ComfyUI's headless execution
API (init_extra_nodes sync-vs-async, PromptServer/PromptExecutor signatures,
validate_prompt arity) has drifted across releases. We pin a known-good commit
via the git submodule, but the gate on this whole approach is whether the
executor + Impact-Pack initialize cleanly under ZeroGPU's fork model. Run the
smoke test before trusting it:

    ARS_FABULA_COMFY_MODE=embed python comfy_embed.py --smoke
"""
from __future__ import annotations
import os
import sys
import glob
import time
import shutil
from typing import Optional

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the ComfyUI submodule lives. Overridable so Docker/local can point at
# an existing checkout. Models go under <COMFY_DIR>/models/{unet,text_encoders,
# vae,sams,ultralytics}.
COMFY_DIR = os.getenv(
    "ARS_FABULA_COMFY_DIR",
    os.path.join(PROJECT_DIR, "third_party", "comfyui"),
)

# Custom nodes are sibling submodules (git won't nest a submodule inside the
# ComfyUI submodule's working tree), linked into ComfyUI's custom_nodes/ dir at
# init. Each entry is a directory name under CUSTOM_NODES_SRC.
CUSTOM_NODES_SRC = os.getenv(
    "ARS_FABULA_COMFY_NODES_DIR",
    os.path.join(PROJECT_DIR, "third_party", "custom_nodes"),
)
CUSTOM_NODES = ["ComfyUI-GGUF", "comfyui-impact-pack", "ComfyUI-Impact-Subpack"]

# Git sources for runtime staging on the Space. In the repo, ComfyUI + these
# nodes are git submodules under third_party/ (great for local/Docker dev), but
# `hf upload` does NOT materialise submodule contents in a Gradio Space's
# runtime checkout. So on the Space we CLONE them at startup — the same
# self-heal pattern as hf_hub_download for the weights. Commits match the
# Dockerfile pins. (None = default branch HEAD, as the Dockerfile does.)
COMFY_GIT = (os.getenv("ARS_FABULA_COMFY_REPO",
                       "https://github.com/comfyanonymous/ComfyUI"),
             "b6332446")
CUSTOM_NODE_GIT = {
    "ComfyUI-GGUF": ("https://github.com/city96/ComfyUI-GGUF", None),
    "comfyui-impact-pack": ("https://github.com/ltdrdata/ComfyUI-Impact-Pack", None),
    "ComfyUI-Impact-Subpack": (
        "https://github.com/ltdrdata/ComfyUI-Impact-Subpack",
        "50c7b71a6a224734cc9b21963c6d1926816a97f1"),
}

# @spaces.GPU duration (seconds) for one workflow run. The first call also
# loads the diffusion stack (~6GB GGUF + encoder + VAE + SAM), so it is the
# slowest; later calls reuse the in-RAM model cache. Bump via env if the first
# bake times out.
GPU_DURATION = int(os.getenv("ARS_FABULA_COMFY_GPU_DURATION", "120"))

# Decorator: real @spaces.GPU on Spaces, a no-op everywhere else — same guard
# as model_client so embed mode is testable locally on a real GPU.
try:
    import spaces as _spaces
    _gpu = _spaces.GPU(duration=GPU_DURATION)
except ImportError:
    def _gpu(f):
        return f


_STATE: dict = {"inited": False, "executor": None, "server": None, "error": None}


# ═══════════════════════════════════════════════════════════════════════
# Filesystem helpers (no torch/comfy import — safe to call anytime)
# ═══════════════════════════════════════════════════════════════════════

def _models_root() -> str:
    return os.path.join(COMFY_DIR, "models")


# Required model files keyed by their ComfyUI sub-directory. Mirrors the
# Dockerfile download targets exactly so http (Docker) and embed (ZeroGPU)
# resolve identical filenames.
REQUIRED_MODELS = {
    "unet": ["anima-base-v1.0-Q5_K_M.gguf"],
    "text_encoders": ["qwen_3_06b_base.safetensors"],
    "vae": ["qwen_image_vae.safetensors"],
    "sams": ["sam_vit_b_01ec64.pth"],
    os.path.join("ultralytics", "bbox"): ["face_yolov8n.pt"],
}


def _missing_models() -> list[str]:
    """Return relative paths of any required model file not yet staged."""
    missing = []
    for subdir, files in REQUIRED_MODELS.items():
        for fn in files:
            if not os.path.exists(os.path.join(_models_root(), subdir, fn)):
                missing.append(os.path.join(subdir, fn))
    return missing


def input_dir() -> str:
    """ComfyUI input folder (LoadImage reads from here)."""
    d = os.path.join(COMFY_DIR, "input")
    os.makedirs(d, exist_ok=True)
    return d


def output_dir() -> str:
    """ComfyUI output folder (SaveImage writes here)."""
    d = os.path.join(COMFY_DIR, "output")
    os.makedirs(d, exist_ok=True)
    return d


def health() -> bool:
    """True if embed mode can plausibly run: the ComfyUI checkout exists and
    all required model files are staged. Deliberately cheap — does NOT import
    torch or ComfyUI (that happens lazily under @spaces.GPU on first run).

    Logs the specific reason on failure (the gate that decides curated-vs-live
    on the Space), so a fallback is never silent/mysterious in the logs."""
    if not os.path.isdir(COMFY_DIR) or not os.path.exists(
            os.path.join(COMFY_DIR, "nodes.py")):
        print(f"[comfy-embed] health: ComfyUI checkout not found at {COMFY_DIR}")
        try:
            tp = os.path.join(PROJECT_DIR, "third_party")
            print(f"[comfy-embed]   PROJECT_DIR={PROJECT_DIR} listdir="
                  f"{sorted(os.listdir(PROJECT_DIR))[:40]}")
            print(f"[comfy-embed]   third_party exists={os.path.isdir(tp)}"
                  + (f" listdir={sorted(os.listdir(tp))}" if os.path.isdir(tp) else ""))
            cdir = os.path.join(tp, "comfyui")
            if os.path.isdir(cdir):
                print(f"[comfy-embed]   comfyui listdir(20)={sorted(os.listdir(cdir))[:20]}")
        except Exception as e:
            print(f"[comfy-embed]   (listdir failed: {e})")
        return False
    missing = _missing_models()
    if missing:
        print(f"[comfy-embed] health: {len(missing)} model file(s) not staged: "
              f"{missing}")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# Model staging (Gradio SDK Space has no Dockerfile to bake weights in)
# ═══════════════════════════════════════════════════════════════════════

def _git_clone(url: str, dest: str, commit: Optional[str] = None) -> bool:
    """Clone `url` into `dest` (merging into an existing dir without clobbering,
    so a pre-existing models/ survives), optionally checking out `commit`.
    Idempotent: a no-op if `dest` already looks populated."""
    import subprocess
    # "Populated" means the source we NEED is present — not merely a .git
    # pointer (HF may leave an empty submodule mount with just a .git file).
    if os.path.isdir(dest) and any(
            os.path.exists(os.path.join(dest, m)) for m in ("nodes.py", "__init__.py")):
        return True
    tmp = dest.rstrip("/") + "__clone"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        subprocess.run(["git", "clone", "--filter=blob:none", "--quiet", url, tmp],
                       check=True, timeout=600)
        if commit:
            subprocess.run(["git", "-C", tmp, "checkout", "--quiet", commit],
                           check=True, timeout=120)
    except Exception as e:
        print(f"[comfy-embed] clone failed {url}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    # Merge tmp/* into dest (keep any files dest already has, e.g. models/).
    os.makedirs(dest, exist_ok=True)
    for entry in os.listdir(tmp):
        src_e, dst_e = os.path.join(tmp, entry), os.path.join(dest, entry)
        if not os.path.exists(dst_e):
            shutil.move(src_e, dst_e)
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def stage_comfyui() -> bool:
    """Ensure the ComfyUI source + custom nodes exist on disk (clone if missing).

    Needed on the Gradio Space, where the third_party submodules don't
    materialise from `hf upload`. No-op locally / on Docker where the checkout
    (or ARS_FABULA_COMFY_DIR) already has them. Returns True if nodes.py is
    present afterward."""
    if not os.path.exists(os.path.join(COMFY_DIR, "nodes.py")):
        print(f"[comfy-embed] cloning ComfyUI {COMFY_GIT[1] or 'HEAD'} → {COMFY_DIR}")
        os.makedirs(os.path.dirname(COMFY_DIR) or ".", exist_ok=True)
        _git_clone(COMFY_GIT[0], COMFY_DIR, COMFY_GIT[1])
    os.makedirs(CUSTOM_NODES_SRC, exist_ok=True)
    for name, (url, commit) in CUSTOM_NODE_GIT.items():
        dest = os.path.join(CUSTOM_NODES_SRC, name)
        if not os.path.exists(os.path.join(dest, "__init__.py")):
            print(f"[comfy-embed] cloning custom node {name}")
            _git_clone(url, dest, commit)
    ok = os.path.exists(os.path.join(COMFY_DIR, "nodes.py"))
    print(f"[comfy-embed] ComfyUI source {'ready' if ok else 'MISSING'}")
    return ok


def stage_models() -> bool:
    """Download the Anima/SAM/YOLO weights into ComfyUI's model dirs.

    Idempotent: skips files already present. Returns True if every required
    file is present afterwards. Safe to call at Space startup; on the Docker
    path (http mode) it is never called.
    """
    missing = _missing_models()
    if not missing:
        print("[comfy-embed] all models already staged")
        return True
    print(f"[comfy-embed] staging {len(missing)} model file(s)…")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"[comfy-embed] huggingface_hub unavailable ({e}); cannot stage")
        return False

    # (repo_id, repo_filename, dest_subdir, dest_filename)
    HF_FILES = [
        ("Abiray/Anima-base-v1.0-GGUF", "anima-base-v1.0-Q5_K_M.gguf",
         "unet", "anima-base-v1.0-Q5_K_M.gguf"),
        ("circlestone-labs/Anima",
         "split_files/text_encoders/qwen_3_06b_base.safetensors",
         "text_encoders", "qwen_3_06b_base.safetensors"),
        ("circlestone-labs/Anima",
         "split_files/vae/qwen_image_vae.safetensors",
         "vae", "qwen_image_vae.safetensors"),
        ("Bingsu/adetailer", "face_yolov8n.pt",
         os.path.join("ultralytics", "bbox"), "face_yolov8n.pt"),
    ]
    for repo_id, repo_fn, sub, dest_fn in HF_FILES:
        dest = os.path.join(_models_root(), sub, dest_fn)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            print(f"[comfy-embed]   {repo_id} :: {repo_fn}")
            src = hf_hub_download(repo_id=repo_id, filename=repo_fn)
            shutil.copyfile(src, dest)
        except Exception as e:
            print(f"[comfy-embed]   FAILED {repo_id}/{repo_fn}: {e}")

    # SAM is hosted on fbaipublicfiles (no HF repo) — same URL as the Dockerfile.
    sam_dest = os.path.join(_models_root(), "sams", "sam_vit_b_01ec64.pth")
    if not os.path.exists(sam_dest):
        os.makedirs(os.path.dirname(sam_dest), exist_ok=True)
        url = ("https://dl.fbaipublicfiles.com/segment_anything/"
               "sam_vit_b_01ec64.pth")
        try:
            print(f"[comfy-embed]   {url}")
            import urllib.request
            urllib.request.urlretrieve(url, sam_dest)
        except Exception as e:
            print(f"[comfy-embed]   FAILED SAM download: {e}")

    still_missing = _missing_models()
    if still_missing:
        print(f"[comfy-embed] still missing after staging: {still_missing}")
        return False
    print("[comfy-embed] model staging complete")
    return True


# ═══════════════════════════════════════════════════════════════════════
# In-process ComfyUI executor
# ═══════════════════════════════════════════════════════════════════════

def _ensure() -> bool:
    """Import ComfyUI headless and build a single PromptExecutor, once.

    Returns True on success. Records the error on _STATE['error'] and returns
    False on failure (so callers degrade gracefully instead of crashing the
    casting beat). All CUDA-touching work is deferred to executor.execute,
    which only runs inside run_workflow's @spaces.GPU window.
    """
    if _STATE["inited"]:
        return _STATE["executor"] is not None
    _STATE["inited"] = True

    if not os.path.isdir(COMFY_DIR):
        _STATE["error"] = f"ComfyUI checkout missing at {COMFY_DIR} " \
                          f"(did the git submodule init?)"
        print(f"[comfy-embed] {_STATE['error']}")
        return False

    if COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)

    _link_custom_nodes()

    # ComfyUI parses sys.argv at import of comfy.cli_args; our app.py has its
    # own argv (the Gradio port). Neutralize argv around the import so ComfyUI
    # doesn't choke on flags it doesn't recognize, then restore.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        import asyncio

        # 0) Pin the real top-level `utils` package into sys.modules NOW, while
        #    COMFY_DIR (which contains the utils/ PACKAGE) is first on sys.path.
        #    Otherwise, once ComfyUI puts its `comfy/` dir on sys.path, a bare
        #    `import utils` resolves to comfy/utils.py — a MODULE, not a package
        #    — and ComfyUI's own `from utils.install_util import …` (in
        #    app/frontend_management, reached via `import server`) blows up with
        #    "'utils' is not a package". Resolving it here caches the correct
        #    package so every later import uses it. Tolerate older layouts.
        try:
            import utils                # noqa: F401  (ComfyUI/utils/ package)
            import utils.install_util   # noqa: F401  (force submodule resolve)
        except Exception as e:
            print(f"[comfy-embed] utils pin skipped ({e})")

        import nodes  # noqa: F401  ComfyUI core node registry
        import execution

        # 1) Build the headless PromptServer FIRST — before loading any custom
        #    nodes. This mirrors ComfyUI's own main.py order and is load-bearing:
        #    importing `server` pulls in app/frontend_management, which does
        #    `from utils.install_util import …`. Custom nodes (e.g. Manager,
        #    AdvancedLivePortrait) inject their own dirs onto sys.path and can
        #    shadow the top-level `utils` package with a bare `utils.py`, after
        #    which that import fails ("'utils' is not a package"). Importing
        #    server while `utils` still resolves to ComfyUI/utils avoids it.
        import server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server_instance = server.PromptServer(loop)
        # Some versions read PromptServer.instance as a class attribute.
        try:
            server.PromptServer.instance = server_instance
        except Exception:
            pass

        # 2) Load custom nodes (ComfyUI-GGUF, Impact-Pack, Impact-Subpack).
        #    init_extra_nodes is sync in older ComfyUI, a coroutine in newer.
        init = getattr(nodes, "init_extra_nodes", None) or \
            getattr(nodes, "init_custom_nodes", None)
        if init is not None:
            res = init()
            if asyncio.iscoroutine(res):
                loop.run_until_complete(res)

        # 3) PromptExecutor signature has varied across versions. Current
        #    ComfyUI: PromptExecutor(server, cache_type, cache_args) — and
        #    execute() reads cache_args["ram"] unconditionally, so cache_args
        #    MUST be a dict with "ram"/"lru" even for the CLASSIC cache (the
        #    default headroom is only used under RAM_PRESSURE). Try the modern
        #    form first, then fall back to older signatures.
        CacheType = getattr(execution, "CacheType", None)
        builders = []
        if CacheType is not None:
            builders.append(lambda: execution.PromptExecutor(
                server_instance, cache_type=CacheType.CLASSIC,
                cache_args={"lru": 0, "ram": 16.0}))
        builders += [
            lambda: execution.PromptExecutor(server_instance),
            lambda: execution.PromptExecutor(server_instance, lru_size=0),
            lambda: execution.PromptExecutor(),
        ]
        executor = None
        for build in builders:
            try:
                executor = build()
                break
            except TypeError:
                continue
        if executor is None:
            raise RuntimeError("could not construct execution.PromptExecutor")

        _STATE["server"] = server_instance
        _STATE["executor"] = executor
        print("[comfy-embed] ComfyUI executor ready (in-process)")
        return True
    except Exception as e:
        import traceback
        _STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[comfy-embed] init failed: {_STATE['error']}")
        traceback.print_exc()
        return False
    finally:
        sys.argv = saved_argv


def _link_custom_nodes() -> None:
    """Make the sibling custom-node submodules visible to ComfyUI by linking
    (or copying, if symlinks are unavailable) each into COMFY_DIR/custom_nodes.

    Idempotent. ComfyUI's init_extra_nodes scans its own custom_nodes/ dir, so
    this is how the GGUF loader / Impact-Pack get registered without nesting
    submodules inside the ComfyUI submodule's working tree.
    """
    dst_root = os.path.join(COMFY_DIR, "custom_nodes")
    os.makedirs(dst_root, exist_ok=True)
    for name in CUSTOM_NODES:
        src = os.path.join(CUSTOM_NODES_SRC, name)
        dst = os.path.join(dst_root, name)
        if not os.path.isdir(src):
            print(f"[comfy-embed] custom node missing: {src} "
                  f"(submodule not initialised?)")
            continue
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        try:
            os.symlink(src, dst, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Windows without privilege / filesystems without symlinks.
            shutil.copytree(src, dst)


def _output_node_ids(workflow: dict) -> list[str]:
    """Node ids whose outputs we want materialized (SaveImage/PreviewImage)."""
    out = [nid for nid, node in workflow.items()
           if node.get("class_type") in ("SaveImage", "PreviewImage")]
    return out or list(workflow.keys())


def _save_prefixes(workflow: dict) -> list[str]:
    """filename_prefix values of the SaveImage nodes — used to locate the
    written files version-independently (we glob the output dir for them)."""
    return [node["inputs"].get("filename_prefix", "")
            for node in workflow.values()
            if node.get("class_type") == "SaveImage"
            and isinstance(node.get("inputs"), dict)]


def _validate(prompt: dict, prompt_id: str):
    """Best-effort validate_prompt across signature variants. Never fatal —
    execute() does its own validation, so this is just for clearer warnings."""
    try:
        import asyncio
        import execution
        vp = getattr(execution, "validate_prompt", None)
        if vp is None:
            return
        # Try signatures newest→oldest. The current form is async:
        #   validate_prompt(prompt_id, prompt, partial_execution_list)
        for args in ((prompt_id, prompt, None), (prompt_id, prompt), (prompt,)):
            try:
                result = vp(*args)
            except TypeError:
                continue
            if asyncio.iscoroutine(result):
                result = asyncio.new_event_loop().run_until_complete(result)
            if isinstance(result, (list, tuple)) and result and not result[0]:
                print(f"[comfy-embed] validate_prompt warning: {result[1:]}")
            return
    except Exception:
        return   # validation is optional; stay quiet


def _execute_one(workflow: dict, client_id: str) -> Optional[dict]:
    """Run ONE workflow on the (already-initialised) executor and return its
    first-image info dict. NOT decorated — call only from inside an @spaces.GPU
    function (run_workflow / run_batch) so the CUDA work is in a GPU window."""
    executor = _STATE["executor"]
    prompt_id = f"{client_id}-{int(time.time() * 1000)}"
    self_outputs = _output_node_ids(workflow)
    prefixes = _save_prefixes(workflow)
    out_dir = output_dir()
    started = time.time()

    _validate(workflow, prompt_id)
    try:
        # execute(prompt, prompt_id, extra_data, execute_outputs) — the stable
        # core signature across the versions we pin.
        executor.execute(workflow, prompt_id, {}, self_outputs)
    except Exception as e:
        import traceback
        print(f"[comfy-embed] execute failed for '{client_id}': "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    # Locate the written file by SaveImage prefix (version-independent —
    # avoids depending on executor.history_result's shape). Pick the newest
    # file written at/after we started executing.
    candidates: list[str] = []
    for prefix in prefixes:
        candidates += glob.glob(os.path.join(out_dir, f"{prefix}*"))
    candidates = [p for p in candidates
                  if os.path.isfile(p) and os.path.getmtime(p) >= started - 1]
    if not candidates:
        print(f"[comfy-embed] no output file found for '{client_id}' "
              f"(prefixes={prefixes})")
        return None
    newest = max(candidates, key=os.path.getmtime)
    return {"filename": os.path.basename(newest), "subfolder": "",
            "type": "output", "_path": newest}


@_gpu
def run_workflow(workflow: dict, client_id: str) -> Optional[dict]:
    """Execute one workflow graph in-process and return its first-image info
    dict (shaped like the HTTP _comfy_wait return), or None. Runs inside one
    @spaces.GPU allocation."""
    if not _ensure():
        return None
    return _execute_one(workflow, client_id)


@_gpu
def run_batch(items: list) -> list:
    """Execute several workflows in ONE @spaces.GPU allocation and return a
    list of their first-image info dicts (None for any that failed), in order.

    This is the ZeroGPU cost win: each GPU call carries ~tens of seconds of
    allocation/attach overhead, so baking a character's whole expression set in
    one call instead of one-call-per-image collapses that overhead. `items` is
    a list of (workflow, client_id) tuples."""
    if not _ensure():
        return [None] * len(items)
    results = []
    for workflow, client_id in items:
        try:
            results.append(_execute_one(workflow, client_id))
        except Exception as e:
            print(f"[comfy-embed] batch item '{client_id}' failed: {e}")
            results.append(None)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Smoke test — the gate on the whole approach. Run on the target Space.
# ═══════════════════════════════════════════════════════════════════════

def _smoke() -> int:
    """Import ComfyUI headless and run a tiny base txt2img once. Prints PASS
    or the first failure. This is what tells us embed mode is viable under
    ZeroGPU's fork model before we flip the live Space."""
    print(f"[smoke] COMFY_DIR = {COMFY_DIR}")
    print(f"[smoke] health()  = {health()}")
    miss = _missing_models()
    if miss:
        print(f"[smoke] missing models {miss}; attempting stage_models()…")
        stage_models()
    if not _ensure():
        print(f"[smoke] FAIL — executor init: {_STATE.get('error')}")
        return 1

    # Minimal Anima base workflow (same node trio the pipeline uses).
    wf = {
        "1": {"inputs": {"filename_prefix": "smoke", "images": ["8", 0]},
              "class_type": "SaveImage"},
        "8": {"inputs": {"samples": ["19", 0], "vae": ["15", 0]},
              "class_type": "VAEDecode"},
        "11": {"inputs": {"text": "1girl, standing, absurdres", "clip": ["45", 0]},
               "class_type": "CLIPTextEncode"},
        "12": {"inputs": {"text": "worst quality", "clip": ["45", 0]},
               "class_type": "CLIPTextEncode"},
        "15": {"inputs": {"vae_name": "qwen_image_vae.safetensors"},
               "class_type": "VAELoader"},
        "19": {"inputs": {"seed": 42, "steps": 8, "cfg": 4.0,
                          "sampler_name": "euler", "scheduler": "simple",
                          "denoise": 1.0, "model": ["44", 0],
                          "positive": ["11", 0], "negative": ["12", 0],
                          "latent_image": ["28", 0]},
               "class_type": "KSampler"},
        "28": {"inputs": {"width": 384, "height": 384, "batch_size": 1},
               "class_type": "EmptyLatentImage"},
        "44": {"inputs": {"unet_name": "anima-base-v1.0-Q5_K_M.gguf"},
               "class_type": "UnetLoaderGGUF"},
        "45": {"inputs": {"clip_name": "qwen_3_06b_base.safetensors",
                          "type": "stable_diffusion", "device": "default"},
               "class_type": "CLIPLoader"},
    }
    info = run_workflow(wf, "smoke")
    if info and os.path.exists(info.get("_path", "")):
        print(f"[smoke] PASS — wrote {info['_path']}")
        return 0
    print("[smoke] FAIL — no image produced")
    return 1


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        raise SystemExit(_smoke())
    print("comfy_embed: in-process ComfyUI transport. Use --smoke to test.")
    print(f"  COMFY_DIR={COMFY_DIR}")
    print(f"  health()={health()}  missing={_missing_models()}")
