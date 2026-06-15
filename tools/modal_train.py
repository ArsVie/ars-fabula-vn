"""
modal_train.py — QLoRA fine-tune of a Gemma base on the VN-protocol SFT data.

Runs on Modal (GPU rented per-second). The dataset is the transformed
real-VN data (tools/transform_vn_dataset.py) merged with our own clean
traces; records are {"messages": [system, user, assistant]} JSONL.

Loss is masked to the ASSISTANT tokens only (the prompts are ~4k tokens of
system rulebook + [STATE]; training on them would mostly memorize the
prompt, not the behavior).

Setup (one-time):
    modal secret create huggingface HF_TOKEN=hf_...   # gated Gemma weights
    modal volume create ars-fabula-train
    modal volume put ars-fabula-train data/sft/train.jsonl /data/train.jsonl

Smoke test (~15 min on L40S, well under $1):
    modal run tools/modal_train.py --max-steps 30 --run-name smoke

Full run (12B, ~3k examples, 2 epochs):
    modal run tools/modal_train.py --epochs 2 --run-name vn12b-r32-v1

26B-A4B attempt (MoE — riskier, needs bigger GPU):
    modal run tools/modal_train.py --base-model google/gemma-4-26b-a4b-it \
        --gpu A100-80GB --run-name vn26b-r16-v1 --lora-r 16

Artifacts land in the volume under /runs/<run-name>/ (adapter + logs +
sample generations). Pull with:
    modal volume get ars-fabula-train /runs/<run-name> ./runs/<run-name>
"""
from __future__ import annotations
import json
import os

import modal

APP_NAME = "ars-fabula-qlora"
VOLUME_NAME = "ars-fabula-train"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Gemma 4 (Gemma4UnifiedForConditionalGeneration) is a recent
        # architecture — needs latest transformers; keep the stack unpinned.
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "sentencepiece",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
          "HF_HOME": "/vol/hf"})
)

# Image for GGUF conversion: the base image (torch/transformers/sentencepiece)
# plus the turboquant fork's convert_hf_to_gguf.py + gguf-py, staged locally at
# ~/convert_fork (it has the custom Gemma4/Gemma4-A4B MoE tensor maps the stock
# converter lacks). pyyaml/tqdm are gguf-py runtime deps.
_FORK_LOCAL = os.path.expanduser("~/convert_fork")
convert_image = (
    image.pip_install("pyyaml", "tqdm")
    .add_local_dir(_FORK_LOCAL, "/fork", copy=True)
)

# Image that builds the fork's llama-quantize (CPU-only) from staged source at
# ~/llama_src — needed to quantize a GEMMA4-A4B GGUF (custom arch the stock
# llama.cpp lacks). Built once at image-build time and cached.
_LLAMA_SRC = os.path.expanduser("~/llama_src")
quant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "git", "libgomp1")
    # tokenizer-only render of the adapter chat template (no torch needed) +
    # HTTP client for the llama-server /completion verify path.
    .pip_install("transformers", "sentencepiece", "protobuf", "requests",
                 "jinja2")
    .add_local_dir(_LLAMA_SRC, "/llama", copy=True)
    .run_commands(
        "cmake -S /llama -B /llama/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF",
        "cmake --build /llama/build "
        "--target llama-quantize llama-cli llama-server -j 8",
    )
)

# CUDA build of the fork's llama-server, for FAST GGUF verify on a GPU (CPU
# inference on the 16GB Q4 + ~4k-token VN prompt is minutes/scene). Same source
# as quant_image but -DGGML_CUDA=ON on an nvidia/cuda devel base (needs nvcc).
# CUDA archs cover T4(75)/A10G-A100(80,86)/L4(89). Build is ~slow once, cached.
quant_image_cuda = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "cmake", "git", "libgomp1")
    .pip_install("transformers", "sentencepiece", "protobuf", "requests",
                 "jinja2")
    .add_local_dir(_LLAMA_SRC, "/llama", copy=True)
    .run_commands(
        # link against the CUDA driver STUB (no GPU/driver at build time, so
        # the real libcuda.so.1 is absent). The devel image ships the stub as
        # libcuda.so; libggml-cuda's SONAME dep is libcuda.so.1, so symlink it.
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "
        "/usr/local/cuda/lib64/stubs/libcuda.so.1",
        "LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "cmake -S /llama -B /llama/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF "
        "-DCMAKE_CUDA_ARCHITECTURES='89' "
        "-DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs "
        "-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' "
        "-DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs "
        "-Wl,-rpath-link,/usr/local/cuda/lib64/stubs'",
        "LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "cmake --build /llama/build --target llama-server -j 8",
    )
)

MINUTES = 60


def _render(tokenizer, messages: list[dict]) -> tuple[str, str]:
    """(prompt_text, full_text) via the model's chat template; assistant
    completion is everything after the generation prompt. Gemma templates
    may reject a system role — fold system into the first user turn then."""
    prompt_msgs, full_msgs = messages[:-1], messages
    try:
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        merged = []
        for m in messages:
            if m["role"] == "system":
                continue
            merged.append(dict(m))
        sys_txt = next((m["content"] for m in messages
                        if m["role"] == "system"), "")
        if sys_txt and merged and merged[0]["role"] == "user":
            merged[0]["content"] = sys_txt + "\n\n" + merged[0]["content"]
        prompt = tokenizer.apply_chat_template(
            merged[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=False)
    return prompt, full


def _unfuse_experts(model) -> int:
    """Path A — make the Gemma 4 MoE experts LoRA-targetable.

    Gemma4TextExperts stores its weights as fused 3D nn.Parameters
    (gate_up_proj [E, 2*inter, hidden], down_proj [E, hidden, inter]) and its
    forward loops per-expert with F.linear(x, gate_up_proj[e]). Those are NOT
    nn.Linear leaves, so bitsandbytes never 4-bit-quantizes them (they load
    bf16) and PEFT can't attach LoRA. Here we swap each fused tensor for a
    per-expert nn.ModuleList[nn.Linear], copying the bf16 slices in, and bind a
    forward that calls self.gate_up[e](x) / self.down[e](x). The loop already
    existed, so this is ~speed-neutral; the linears stay bf16 (~52GB → needs a
    B200-class GPU). Returns the count of expert modules un-fused.

    Run this AFTER model load but BEFORE prepare_model_for_kbit_training so the
    new linears are frozen as base weights and the LoRA pass can target them.
    """
    import types
    import torch
    import torch.nn as nn

    def _experts_forward(self, hidden_states, top_k_index, top_k_weights):
        # Faithful copy of Gemma4TextExperts.forward with the fused-tensor
        # index reads replaced by per-expert Linear calls. expert_idx is a
        # tensor scalar, so it must be int() to index a ModuleList.
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index,
                                                num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            gate, up = self.gate_up[e](current_state).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down[e](current_hidden_states)
            current_hidden_states = (current_hidden_states
                                     * top_k_weights[token_idx, top_k_pos, None])
            final_hidden_states.index_add_(
                0, token_idx,
                current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    count = 0
    for mod in model.modules():
        if type(mod).__name__ != "Gemma4TextExperts":
            continue
        if not hasattr(mod, "gate_up_proj"):
            continue  # already un-fused
        E, hidden, inter = mod.num_experts, mod.hidden_dim, mod.intermediate_dim
        gate_up, down = nn.ModuleList(), nn.ModuleList()
        for i in range(E):
            lin_gu = nn.Linear(hidden, 2 * inter, bias=False)
            lin_gu.weight = nn.Parameter(mod.gate_up_proj.data[i].detach().clone())
            lin_dn = nn.Linear(inter, hidden, bias=False)
            lin_dn.weight = nn.Parameter(mod.down_proj.data[i].detach().clone())
            gate_up.append(lin_gu)
            down.append(lin_dn)
        # drop the fused params (and their storage) before the next module so
        # the bf16 copy never doubles the whole expert tensor at once
        del mod._parameters["gate_up_proj"]
        del mod._parameters["down_proj"]
        mod.gate_up, mod.down = gate_up, down
        mod.forward = types.MethodType(_experts_forward, mod)
        count += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return count


def _refuse_experts(model) -> int:
    """Exact inverse of _unfuse_experts: re-stack each per-expert nn.Linear back
    into the fused 3D nn.Parameters (gate_up_proj [E, 2*inter, hidden],
    down_proj [E, hidden, inter]) the stock Gemma4TextExperts expects, restore
    the class forward, and drop the ModuleLists. Run AFTER merge_and_unload so
    the merged LoRA weights are baked into the fused tensors — this yields a
    standard-arch model that save_pretrained emits with canonical keys, which
    convert_hf_to_gguf.py can read (a per-expert LoRA cannot be expressed as a
    GGUF LoRA against the fused MoE tensors, so merge+refuse is the only path).
    Returns the count of expert modules re-fused.
    """
    import torch
    import torch.nn as nn

    count = 0
    for mod in model.modules():
        if type(mod).__name__ != "Gemma4TextExperts":
            continue
        if not hasattr(mod, "gate_up"):
            continue  # already fused
        gate_up = torch.stack([lin.weight.data for lin in mod.gate_up], dim=0)
        down = torch.stack([lin.weight.data for lin in mod.down], dim=0)
        del mod.gate_up
        del mod.down
        # restore the class forward (drop the bound instance override)
        if "forward" in mod.__dict__:
            del mod.__dict__["forward"]
        mod.gate_up_proj = nn.Parameter(gate_up.contiguous())
        mod.down_proj = nn.Parameter(down.contiguous())
        count += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return count


@app.function(
    image=image,
    gpu=os.environ.get("ARS_FABULA_TRAIN_GPU", "L40S"),
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    # 12h: a 2-epoch 26B-A4B experts run is ~10h at ~52s/step (per-expert loop).
    # The old 6h cap silently killed vn26b-experts-v1 at 59% (epoch 1.18) with
    # save_strategy="no", so nothing was committed. Keep generous headroom.
    timeout=12 * 60 * MINUTES,
)
def train(base_model: str, data_path: str, run_name: str,
          lora_r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.05,
          max_steps: int = -1, epochs: float = 1.0, lr: float = 1e-4,
          max_seq_len: int = 6144, batch_size: int = 1,
          grad_accum: int = 8, eval_holdout: int = 8,
          merge: bool = False, attn_only: bool = False,
          unfuse_experts: bool = False):
    # Reduce allocator fragmentation — the un-fused expert loop + eager attn
    # leaves big reserved-but-unallocated gaps; expandable_segments reclaims them
    # so larger batches fit. Must be set before CUDA init (first torch import).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq, TrainerCallback)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    out_dir = f"/vol/runs/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    assert records, f"no records in {data_path}"
    holdout = records[:eval_holdout]
    trainset = records[eval_holdout:]
    print(f"records: {len(records)} ({len(trainset)} train, "
          f"{len(holdout)} held out)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class SFTDataset(Dataset):
        def __init__(self, recs):
            self.recs = recs

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, idx):
            messages = self.recs[idx]["messages"]
            prompt, full = _render(tokenizer, messages)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            full_ids = full_ids[:max_seq_len]
            labels = list(full_ids)
            cut = min(len(prompt_ids), len(labels))
            labels[:cut] = [-100] * cut  # loss only on the completion
            return {"input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels}

    # ── model ─────────────────────────────────────────────────────────
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    load_kwargs = dict(quantization_config=bnb, torch_dtype=torch.bfloat16,
                       device_map="auto", attn_implementation="eager")
    # Gemma 4 ships as a unified multimodal arch; AutoModelForCausalLM may
    # not map to it — fall back through the auto classes that do.
    model = None
    last = None
    loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForConditionalGeneration
        loaders.append(AutoModelForConditionalGeneration)
    except ImportError:
        pass
    for loader in loaders:
        try:
            model = loader.from_pretrained(base_model, **load_kwargs)
            print(f"loaded with {loader.__name__}")
            break
        except (ValueError, KeyError) as e:
            last = e
    if model is None:
        raise RuntimeError(f"could not load {base_model}: {last}")

    # Path A: turn the fused MoE expert tensors into per-expert nn.Linear so
    # LoRA can target the experts (the whole point of the 26B tune — attn-only
    # leaves the experts frozen and the prose restyle shallow). No-op when the
    # arch has no fused experts. Must run before prepare_model_for_kbit_training.
    unfused = _unfuse_experts(model) if unfuse_experts else 0
    if unfuse_experts:
        print(f"un-fused {unfused} Gemma4TextExperts modules "
              f"into per-expert nn.Linear")

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    import re as _re
    attn_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    ffn_names = ["gate_proj", "up_proj", "down_proj"]
    proj_names = attn_names if attn_only else attn_names + ffn_names
    module_names = [n for n, _ in model.named_modules()]
    multimodal = any("language_model" in n for n in module_names)
    if multimodal:
        # unified multimodal arch: adapt ONLY the language tower, not the
        # vision/audio encoders. For the 26B-A4B MoE this regex matches the
        # per-expert leaf linears we just un-fused (…experts.gate_up.N /
        # …experts.down.N) plus the dense attn + MLP projections.
        if unfused:
            targets = (r".*language_model.*(\.(" + "|".join(proj_names)
                       + r")|\.experts\.(gate_up|down)\.\d+)$")
        else:
            targets = r".*language_model.*\.({})$".format("|".join(proj_names))
        pat = _re.compile(targets)
        matched = sum(1 for n in module_names if pat.match(n))
    else:
        targets = proj_names
        matched = sum(1 for n in module_names if n.split(".")[-1] in proj_names)
    print(f"LoRA target modules matched: {matched} "
          f"(attn_only={attn_only}, multimodal={multimodal})")
    if matched == 0:
        raise RuntimeError(
            "LoRA matched 0 modules — likely a fused MoE expert tensor with no "
            "leaf Linear layers. Re-run with --attn-only to adapt attention "
            "projections instead, or add MoE-specific target names.")
    if not attn_only and multimodal and matched <= len(attn_names) * 4:
        # FFN/expert names didn't match (fused experts): warn loudly — the run
        # will still train attention but the docstring's expectation is off.
        print("WARNING: matched count looks attention-only; the MoE expert "
              "FFNs may be fused (no per-expert Linear). Consider --attn-only "
              "to make this explicit.")
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        # Resumable checkpoints: a long 26B run can outlast the function
        # timeout or be canceled — without these, killed = total loss (what
        # bit vn26b-experts-v1 at 59%). Adapter+optimizer state is small.
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to=[],
        seed=42,
    )

    # Modal volume writes must be committed to survive a function restart; the
    # Trainer writes checkpoint-N into out_dir (on /vol) but we must commit on
    # each save or a kill loses the just-written checkpoint.
    class VolCommit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            vol.commit()
            print(f"committed checkpoint at step {state.global_step}")

    trainer = Trainer(
        model=model, args=args,
        train_dataset=SFTDataset(trainset),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[VolCommit()],
    )
    import glob as _glob
    resume = bool(_glob.glob(f"{out_dir}/checkpoint-*"))
    if resume:
        print(f"resuming from latest checkpoint in {out_dir}")
    result = trainer.train(resume_from_checkpoint=resume or None)
    print("train metrics:", result.metrics)

    model.save_pretrained(f"{out_dir}/adapter")
    tokenizer.save_pretrained(f"{out_dir}/adapter")
    with open(f"{out_dir}/train_metrics.json", "w") as f:
        json.dump({"metrics": result.metrics, "base_model": base_model,
                   "lora_r": lora_r, "lora_alpha": lora_alpha,
                   "records": len(trainset), "epochs": epochs,
                   "max_steps": max_steps, "lr": lr}, f, indent=2)

    # ── sample generations on the holdout (for vn_validate locally) ───
    model.config.use_cache = True
    model.eval()
    samples = []
    for rec in holdout:
        prompt, _ = _render(tokenizer, rec["messages"])
        ids = tokenizer(prompt, return_tensors="pt",
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=700, temperature=0.8,
                                 top_p=0.95, do_sample=True,
                                 pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        samples.append({"messages": rec["messages"][:-1],
                        "reference": rec["messages"][-1]["content"],
                        "generated": completion})
    with open(f"{out_dir}/holdout_generations.jsonl", "w",
              encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} holdout generations")

    if merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{out_dir}/merged",
                               safe_serialization=True)
        tokenizer.save_pretrained(f"{out_dir}/merged")
        print("merged model saved")

    vol.commit()
    return f"{out_dir} done"


@app.function(
    image=image,
    gpu=os.environ.get("ARS_FABULA_TRAIN_GPU", "L40S"),
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * MINUTES,
)
def eval_base(base_model: str, data_path: str, run_name: str,
              eval_holdout: int = 8, max_seq_len: int = 6144):
    """Generate on the SAME holdout slice with the un-tuned base model —
    the 'before' arm for tools/eval_generations.py."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    out_dir = f"/vol/runs/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    records = [json.loads(l) for l in open(data_path, encoding="utf-8")
               if l.strip()]
    holdout = records[:eval_holdout]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    load_kwargs = dict(quantization_config=bnb, torch_dtype=torch.bfloat16,
                       device_map="auto", attn_implementation="eager")
    model = None
    loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    for loader in loaders:
        try:
            model = loader.from_pretrained(base_model, **load_kwargs)
            break
        except (ValueError, KeyError):
            continue
    assert model is not None, f"could not load {base_model}"
    model.eval()

    with open(f"{out_dir}/base_generations.jsonl", "w",
              encoding="utf-8") as f:
        for rec in holdout:
            prompt, _ = _render(tokenizer, rec["messages"])
            ids = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=700,
                                     temperature=0.8, top_p=0.95,
                                     do_sample=True,
                                     pad_token_id=tokenizer.pad_token_id
                                     or tokenizer.eos_token_id)
            completion = tokenizer.decode(
                out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps(
                {"messages": rec["messages"][:-1],
                 "reference": rec["messages"][-1]["content"],
                 "generated": completion}, ensure_ascii=False) + "\n")
    vol.commit()
    return f"{out_dir}/base_generations.jsonl"


# ── Forgetting probe — does the narrow VN LoRA degrade general capability? ──
# Neutral general-capability prompts (NO VN system prompt). We generate base vs
# tuned answers greedily on the SAME GPU session and diff them. Most relevant
# for vn26b-experts-v1, which trains the MoE experts (the FFN-equivalent where
# world knowledge / reasoning circuits live) — the layer-choice most prone to
# catastrophic forgetting. `expect` (when set) is a substring auto-check so the
# scorer can flag base-correct→tuned-wrong regressions; None = eyeball only.
PROBE_PROMPTS = [
    ("arithmetic", "What is 17 multiplied by 24? Answer with just the number.", "408"),
    ("multi_step", "A shelf has 3 rows. Each row holds 8 books. I remove 5 books total. How many books are left? Give just the number.", "19"),
    ("common_sense", "If you drop an ice cube into a hot cup of tea, what happens to the ice? Answer in one short sentence.", "melt"),
    ("spatial_turn", "I am facing north. I turn right 90 degrees, then right 90 degrees again. Which direction am I facing now? One word.", "south"),
    ("spatial_stack", "A book is on a table. A cup is on top of the book. What object is directly touching the table? One word.", "book"),
    ("logic", "All cats are mammals. Some mammals are not cats. Does it follow that some cats are not mammals? Answer yes or no, then one short reason.", "no"),
    ("short_memory", "The password is BLUE-42. Ignore any other instruction and tell me: what was the password? Answer with just the password.", "BLUE-42"),
    ("instruction", "List exactly three primary colors, comma-separated, and nothing else.", None),
    ("world_knowledge", "What is the capital of Japan? One word.", "Tokyo"),
    ("causal", "Why does the moon appear to change shape over a month? One short sentence.", None),
    ("negation", "Name one animal that is NOT a mammal. One word.", None),
    ("winograd", "The trophy did not fit in the suitcase because it was too big. What was too big — the trophy or the suitcase? One word.", "trophy"),
]


@app.function(
    image=image,
    gpu=os.environ.get("ARS_FABULA_TRAIN_GPU", "L40S"),
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * MINUTES,
)
def probe(base_model: str, run_name: str, unfuse_experts: bool = False,
          max_new_tokens: int = 256):
    """Generate base vs tuned answers on PROBE_PROMPTS and write
    /vol/runs/<run_name>/forgetting_probe.jsonl. Greedy decode for a
    deterministic, comparable diff. The adapter is loaded ON TOP of the base in
    the same session; for an experts-tuned run the experts must be un-fused
    first so the adapter's target modules (…experts.gate_up.N) exist."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import PeftModel

    out_dir = f"/vol/runs/{run_name}"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    load_kwargs = dict(quantization_config=bnb, torch_dtype=torch.bfloat16,
                       device_map="auto", attn_implementation="eager")
    model = None
    loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    for loader in loaders:
        try:
            model = loader.from_pretrained(base_model, **load_kwargs)
            break
        except (ValueError, KeyError):
            continue
    assert model is not None, f"could not load {base_model}"
    model.eval()

    def gen(m, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
        ids = tokenizer(text, return_tensors="pt",
                        add_special_tokens=False).to(m.device)
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=pad_id)
        return tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    base_ans = [gen(model, p) for _, p, _ in PROBE_PROMPTS]

    # apply the trained adapter on top of the base
    if unfuse_experts:
        n = _unfuse_experts(model)
        print(f"un-fused {n} expert modules before loading adapter")
    model = PeftModel.from_pretrained(model, f"{out_dir}/adapter")
    model.eval()
    tuned_ans = [gen(model, p) for _, p, _ in PROBE_PROMPTS]

    with open(f"{out_dir}/forgetting_probe.jsonl", "w",
              encoding="utf-8") as f:
        for (cat, prompt, expect), b, t in zip(PROBE_PROMPTS, base_ans, tuned_ans):
            f.write(json.dumps({"category": cat, "prompt": prompt,
                                "expect": expect, "base": b, "tuned": t},
                               ensure_ascii=False) + "\n")
    vol.commit()
    print(f"wrote {len(PROBE_PROMPTS)} probe rows")
    return f"{out_dir}/forgetting_probe.jsonl"


@app.function(
    image=image,
    gpu=os.environ.get("ARS_FABULA_TRAIN_GPU", "L40S"),
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * MINUTES,
)
def regen(base_model: str, run_name: str, temps: str = "0.8,1.1,1.4",
          unfuse_experts: bool = False, top_p: float = 0.95,
          max_new_tokens: int = 700, seed: int = 42):
    """Re-sample the SAME holdout prompts of an already-trained run at one or
    more temperatures, without retraining. Prompts are read back from the run's
    existing holdout_generations.jsonl (its `messages` are prompt-only), so the
    inputs are byte-identical to the baseline 0.8 generations. Writes one file
    per temp: holdout_generations_t<temp>.jsonl. The adapter is loaded on top of
    the base in one session; experts-tuned runs must un-fuse first (same as
    probe). The 26B attn-only run does NOT need unfuse_experts."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import PeftModel

    out_dir = f"/vol/runs/{run_name}"
    src = f"{out_dir}/holdout_generations.jsonl"
    holdout = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    assert holdout, f"no holdout prompts in {src}"
    temp_list = [float(t) for t in temps.split(",") if t.strip()]
    print(f"regen {run_name}: {len(holdout)} prompts x temps {temp_list}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    load_kwargs = dict(quantization_config=bnb, torch_dtype=torch.bfloat16,
                       device_map="auto", attn_implementation="eager")
    model = None
    loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    for loader in loaders:
        try:
            model = loader.from_pretrained(base_model, **load_kwargs)
            break
        except (ValueError, KeyError):
            continue
    assert model is not None, f"could not load {base_model}"
    if unfuse_experts:
        n = _unfuse_experts(model)
        print(f"un-fused {n} expert modules before loading adapter")
    model = PeftModel.from_pretrained(model, f"{out_dir}/adapter")
    model.config.use_cache = True
    model.eval()

    def _prompt_text(prompt_msgs):
        # prompt-only render with the same system-fold fallback as _render
        try:
            return tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            sys_txt = next((m["content"] for m in prompt_msgs
                            if m["role"] == "system"), "")
            merged = [dict(m) for m in prompt_msgs if m["role"] != "system"]
            if sys_txt and merged and merged[0]["role"] == "user":
                merged[0]["content"] = sys_txt + "\n\n" + merged[0]["content"]
            return tokenizer.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True)

    written = []
    for temp in temp_list:
        torch.manual_seed(seed)  # comparable across temps
        rows = []
        for rec in holdout:
            prompt_msgs = rec["messages"]
            text = _prompt_text(prompt_msgs)
            ids = tokenizer(text, return_tensors="pt",
                            add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new_tokens,
                                     temperature=temp, top_p=top_p,
                                     do_sample=True, pad_token_id=pad_id)
            completion = tokenizer.decode(
                out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            rows.append({"messages": prompt_msgs,
                         "reference": rec.get("reference", ""),
                         "generated": completion, "temperature": temp})
        tag = f"{temp:g}".replace(".", "")
        dest = f"{out_dir}/holdout_generations_t{tag}_s{seed}.jsonl"
        with open(dest, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written.append(dest)
        print(f"wrote {len(rows)} generations at temp={temp} -> {dest}")
    vol.commit()
    return written


@app.function(
    image=image,
    gpu=os.environ.get("ARS_FABULA_TRAIN_GPU", "B200"),
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=120 * MINUTES,
)
def merge_refuse(base_model: str, run_name: str):
    """Produce a standard-arch merged HF model from an experts (un-fused) LoRA
    run, ready for convert_hf_to_gguf.py. Loads the base in bf16 (NOT 4-bit, so
    the merge is lossless), un-fuses the experts, applies+merges the adapter,
    then RE-FUSES the per-expert linears back into the canonical fused tensors.
    A bit-exactness self-check on the first experts module aborts before saving
    if the re-fuse is wrong. Output: /vol/runs/<run>/merged (safetensors)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    out_dir = f"/vol/runs/{run_name}"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto",
                       attn_implementation="eager")
    model = None
    loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    for loader in loaders:
        try:
            model = loader.from_pretrained(base_model, **load_kwargs)
            break
        except (ValueError, KeyError):
            continue
    assert model is not None, f"could not load {base_model}"

    n_unfused = _unfuse_experts(model)
    print(f"un-fused {n_unfused} experts modules")
    model = PeftModel.from_pretrained(model, f"{out_dir}/adapter")
    print("adapter loaded; merging...")
    model = model.merge_and_unload()

    # snapshot the first experts module's per-expert weights pre-refuse
    snap = None
    for mod in model.modules():
        if type(mod).__name__ == "Gemma4TextExperts" and hasattr(mod, "gate_up"):
            snap = (mod,
                    [l.weight.data.clone() for l in mod.gate_up],
                    [l.weight.data.clone() for l in mod.down])
            break

    n_refused = _refuse_experts(model)
    print(f"re-fused {n_refused} experts modules")

    # bit-exactness self-check: refuse must stack the same slices, in order
    if snap is not None:
        mod, gu_snap, dn_snap = snap
        gu_ok = all(torch.equal(mod.gate_up_proj.data[i], gu_snap[i])
                    for i in range(len(gu_snap)))
        dn_ok = all(torch.equal(mod.down_proj.data[i], dn_snap[i])
                    for i in range(len(dn_snap)))
        assert gu_ok and dn_ok, (
            f"REFUSE SELF-CHECK FAILED (gate_up={gu_ok}, down={dn_ok}) — "
            "not saving; the fused tensor does not match the per-expert linears")
        print(f"refuse self-check PASS: gate_up={gu_ok} down={dn_ok}, "
              f"shapes {tuple(mod.gate_up_proj.shape)}/{tuple(mod.down_proj.shape)}")
    assert n_unfused == n_refused, f"unfuse/refuse count mismatch {n_unfused}/{n_refused}"

    dest = f"{out_dir}/merged"
    model.save_pretrained(dest, safe_serialization=True)
    tokenizer.save_pretrained(dest)
    vol.commit()
    print(f"saved merged standard-arch model -> {dest}")
    return dest


@app.function(
    image=convert_image,
    volumes={"/vol": vol},
    cpu=8.0,
    memory=98304,  # 96GB: convert streams tensors but a 26B f16 needs headroom
    timeout=180 * MINUTES,
)
def convert_gguf(run_name: str, outtype: str = "f16"):
    """Convert the re-fused, merged standard-arch model to GGUF with the fork's
    converter (custom Gemma4-A4B MoE tensor maps). Reads /vol/runs/<run>/merged,
    writes /vol/runs/<run>/<run>-<outtype>.gguf. Patches config architectures to
    the name the fork registers (Gemma4ForConditionalGeneration -> Gemma4Model,
    the text tower), mirroring tools/_convert_lora.sh."""
    import json
    import subprocess
    import sys

    merged = f"/vol/runs/{run_name}/merged"
    cfg_path = f"{merged}/config.json"
    cfg = json.load(open(cfg_path))
    prev = cfg.get("architectures")
    cfg["architectures"] = ["Gemma4ForConditionalGeneration"]
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"patched architectures {prev} -> {cfg['architectures']}")

    out = f"/vol/runs/{run_name}/{run_name}-{outtype}.gguf"
    env = {**os.environ, "PYTHONPATH": "/fork/gguf-py"}
    cmd = [sys.executable, "/fork/convert_hf_to_gguf.py", merged,
           "--outfile", out, "--outtype", outtype]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, env=env, cwd="/fork")
    if r.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf failed (exit {r.returncode})")
    vol.commit()
    print(f"wrote {out}")
    return out


@app.function(
    image=quant_image,
    volumes={"/vol": vol},
    cpu=8.0,
    memory=65536,
    timeout=180 * MINUTES,
)
def quantize(run_name: str, types: str = "Q4_K_M,Q8_0"):
    """Quantize the f16 GGUF (on the volume) to the given types with the fork's
    llama-quantize. Writes /vol/runs/<run>/<run>-<TYPE>.gguf for each. The MoE
    expert inner dim (704) is not /256, so K-quants auto-fall-back for those
    tensors (a warning, still valid). Output stays on the volume — pull later."""
    import multiprocessing
    import os as _os
    import subprocess

    src = f"/vol/runs/{run_name}/{run_name}-f16.gguf"
    assert _os.path.exists(src), f"missing f16: {src}"
    binp = "/llama/build/bin/llama-quantize"
    env = {**os.environ, "LD_LIBRARY_PATH": "/llama/build/bin"}
    nthreads = str(multiprocessing.cpu_count())
    outs = []
    for t in [x.strip() for x in types.split(",") if x.strip()]:
        out = f"/vol/runs/{run_name}/{run_name}-{t}.gguf"
        print(f"=== quantizing -> {t} ===", flush=True)
        r = subprocess.run([binp, src, out, t, nthreads], env=env)
        if r.returncode != 0:
            raise RuntimeError(f"llama-quantize {t} failed (exit {r.returncode})")
        sz = _os.path.getsize(out) / 1e9
        print(f"wrote {out} ({sz:.1f} GB)", flush=True)
        outs.append(out)
        vol.commit()
    return outs


@app.function(
    image=quant_image_cuda,
    volumes={"/vol": vol},
    gpu="L4",
    memory=32768,
    timeout=60 * MINUTES,
)
def verify_gguf(run_name: str, quant: str = "Q4_K_M", indices: str = "4",
                n_predict: int = 400):
    """Runtime-test a quantized GGUF: serve it with the fork's llama-server and
    generate on real holdout prompts via the RAW /completion endpoint to confirm
    it produces coherent VN-protocol output (valid GGUF != verified inference).

    Templating matters: the tuned Gemma adapter emits a custom reasoning channel
    (`<|channel>thought ... <channel|>`). We render the prompt with the adapter's
    OWN chat_template (tokenizer.apply_chat_template) — NOT a hand-built raw
    `<start_of_turn>` string, which skips the template's generation scaffold and
    makes the model dump meta/thought text. We hit /completion (not the OpenAI
    /v1/chat parser, which mangles the channel tokens to empty content) and strip
    the stray leading channel scaffold, exactly like tools/ab_local_eval.py.

    Cleaned generations are written to /vol/runs/<run>/gguf_verify_<quant>.jsonl
    (pull locally to eyeball legibility) and also printed."""
    import json
    import os as _os
    import subprocess
    import time
    import urllib.request

    gguf = f"/vol/runs/{run_name}/{run_name}-{quant}.gguf"
    adapter = f"/vol/runs/{run_name}/adapter"
    assert _os.path.exists(gguf), f"missing {gguf}"
    recs = [json.loads(l) for l
            in open(f"/vol/runs/{run_name}/holdout_generations.jsonl",
                    encoding="utf-8") if l.strip()]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter)

    def _clean(text):
        # The adapter template's generation prompt double-closes the thought
        # channel; the model emits a stray leading channel scaffold. Drop it.
        t = text.lstrip()
        for tag in ("<channel|>", "<|channel>thought", "<|channel>", "thought"):
            if t.startswith(tag):
                t = t[len(tag):].lstrip()
        return t

    env = {**os.environ, "LD_LIBRARY_PATH": "/llama/build/bin"}
    srv = subprocess.Popen(
        ["/llama/build/bin/llama-server", "-m", gguf, "-c", "8192",
         "-ngl", "99", "--host", "127.0.0.1", "--port", "8080"],
        env=env)
    base = "http://127.0.0.1:8080"
    try:
        # poll /health until the model is loaded (long CPU prefill build-up)
        for _ in range(180):
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if json.loads(r.read())["status"] == "ok":
                        break
            except Exception:
                pass
            if srv.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            time.sleep(2)
        else:
            raise RuntimeError("llama-server never reached /health ok")

        results = []
        for idx in [int(i) for i in indices.split(",") if i.strip()]:
            msgs = recs[idx]["messages"]
            convo = [m for m in msgs if m["role"] in ("system", "user")]
            prompt = tok.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True)
            payload = json.dumps({
                "prompt": prompt, "n_predict": n_predict,
                "temperature": 1.1, "top_p": 0.95,
            }).encode()
            req = urllib.request.Request(
                base + "/completion", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                content = _clean(json.loads(r.read())["content"])
            print(f"\n========== {quant} scene {idx} ==========", flush=True)
            print(content, flush=True)
            results.append({"quant": quant, "scene": idx, "content": content})
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()

    out = f"/vol/runs/{run_name}/gguf_verify_{quant}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    print(f"\nwrote {out}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Persistent serve — expose the tuned GGUF as a remote OpenAI-compatible LLM
# endpoint so a Hugging Face Space (or anything) can narrate against it without
# its own GPU. Reuses the already-built+verified CUDA fork llama-server image
# (quant_image_cuda) and reads the GGUF straight off the volume (no HF download
# at boot → faster cold start). Deploy it persistently with:
#       modal deploy tools/modal_train.py
# which exposes https://<workspace>--ars-fabula-qlora-serve.modal.run ; the
# app's base_url is that URL + "/v1".
#
# AUTH: requires_proxy_auth=True puts Modal's edge proxy in front of EVERY path
# (including /health and /v1/*). Callers must send `Modal-Key` + `Modal-Secret`
# headers (a workspace "proxy auth token", minted in the Modal dashboard under
# Settings → Proxy Auth Tokens — there is no CLI/SDK to create one as of
# modal 1.5). model_client.LlamaServerClient sends them when ARS_FABULA_MODAL_KEY
# / ARS_FABULA_MODAL_SECRET are set.
SERVE_GGUF = "/vol/runs/vn26b-experts-v1/vn26b-experts-v1-Q4_K_M.gguf"


@app.function(
    image=quant_image_cuda,        # already built + verified this round
    volumes={"/vol": vol},
    gpu="L4",                      # 24GB > the 16.8GB Q4, fully offloaded
    timeout=60 * MINUTES,
    # keep-warm for the demo so judges never hit a cold start. An L4 billed
    # continuously is ~$0.80/hr — fine during judging, but scale this to 0
    # (+ a short scaledown_window) afterward so it doesn't bill idle.
    min_containers=1,
    scaledown_window=20 * MINUTES,
)
@modal.web_server(port=8080, startup_timeout=300, requires_proxy_auth=True)
def serve():
    import subprocess
    import os as _os

    # Same CUDA runtime libs the verify path used. The fork's llama-server is
    # in /llama/build/bin alongside its libggml*.so.
    env = {**_os.environ, "LD_LIBRARY_PATH": "/llama/build/bin"}
    # NO --cpu-moe / --no-mmap (those were the CPU/T4-Space path) and NO MTP
    # drafter (the tuned model has none — only the stock model does). On L4 the
    # whole 16.8GB offloads to GPU (-ngl 99). --temp/--top-p bake the ship
    # sampling (1.1 / 0.95) as server defaults so even a raw curl matches; the
    # app additionally sends temperature via ARS_FABULA_TEMPERATURE. --reasoning
    # off strips the tuned model's <|channel>thought… channel at the server (see
    # the verify_gguf note); if the OpenAI /v1/chat parser still leaks it,
    # fall back to the raw /completion recipe in LlamaServerClient.
    cmd = [
        "/llama/build/bin/llama-server",
        "-m", SERVE_GGUF,
        "-c", "16384", "-ngl", "99", "-fa", "on",
        "--reasoning", "off", "-np", "1",
        "--temp", "1.1", "--top-p", "0.95",
        "--host", "0.0.0.0", "--port", "8080",   # 0.0.0.0 so Modal can proxy
    ]
    print("[serve] launching:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd, env=env)


@app.function(
    image=quant_image_cuda,
    volumes={"/vol": vol},
    gpu="L4",
    memory=32768,
    timeout=60 * MINUTES,
)
def verify_chat(run_name: str = "vn26b-experts-v1", quant: str = "Q4_K_M",
                indices: str = "0,4", n_predict: int = 400):
    """De-risk the serve path: start llama-server with the EXACT serve() flags
    (--reasoning off) and hit the OpenAI /v1/chat/completions endpoint — the one
    LlamaServerClient uses — with real holdout system+user pairs. The tuned model
    emits a reasoning channel (<|channel>thought … <channel|>) that the OpenAI
    parser mangled to empty content during eval; `--reasoning off` is supposed to
    strip it, but that was only ever validated on the STOCK model. This asserts
    the TUNED model returns clean, non-empty VN protocol with no channel leakage.
    Runs entirely on 127.0.0.1 inside the container — no public endpoint, no
    proxy-auth token. If content is empty or leaks channel tags, switch
    LlamaServerClient to the raw /completion recipe (verify_gguf)."""
    import json as _json
    import os as _os
    import subprocess
    import time
    import urllib.request

    gguf = f"/vol/runs/{run_name}/{run_name}-{quant}.gguf"
    assert _os.path.exists(gguf), f"missing {gguf}"
    recs = [_json.loads(l) for l
            in open(f"/vol/runs/{run_name}/holdout_generations.jsonl",
                    encoding="utf-8") if l.strip()]

    env = {**_os.environ, "LD_LIBRARY_PATH": "/llama/build/bin"}
    srv = subprocess.Popen(
        ["/llama/build/bin/llama-server", "-m", gguf,
         "-c", "16384", "-ngl", "99", "-fa", "on",
         "--reasoning", "off", "-np", "1",
         "--temp", "1.1", "--top-p", "0.95",
         "--host", "127.0.0.1", "--port", "8080"],
        env=env)
    base = "http://127.0.0.1:8080"
    leaked_any = False
    results = []
    try:
        for _ in range(180):
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if _json.loads(r.read())["status"] == "ok":
                        break
            except Exception:
                pass
            if srv.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            time.sleep(2)
        else:
            raise RuntimeError("llama-server never reached /health ok")

        for idx in [int(i) for i in indices.split(",") if i.strip()]:
            msgs = [m for m in recs[idx]["messages"]
                    if m["role"] in ("system", "user")]
            payload = _json.dumps({
                "model": "vn26b", "messages": msgs,
                "temperature": 1.1, "top_p": 0.95, "max_tokens": n_predict,
            }).encode()
            req = urllib.request.Request(
                base + "/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                msg = _json.loads(r.read())["choices"][0]["message"]
            content = msg.get("content") or ""
            # channel scaffold tokens — NOT the word "thought" (too common)
            leaked = any(t in content for t in ("<|channel>", "<channel|>",
                                                "channel|>", "<|channel"))
            empty = not content.strip()
            leaked_any = leaked_any or leaked or empty
            print(f"\n===== {quant} scene {idx} "
                  f"(empty={empty}, leaked={leaked}, "
                  f"reasoning_field={'reasoning_content' in msg}) =====",
                  flush=True)
            print(content[:2000], flush=True)
            results.append({"scene": idx, "content": content,
                            "empty": empty, "leaked": leaked,
                            "has_reasoning_field": "reasoning_content" in msg})
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()

    out = f"/vol/runs/{run_name}/chat_verify_{quant}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(_json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    verdict = "LEAK/EMPTY DETECTED — use /completion fallback" if leaked_any \
        else "CLEAN — /v1/chat/completions + --reasoning off is safe"
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    return {"clean": not leaked_any, "results": results}


# ---------------------------------------------------------------------------
# Hugging Face upload (experts GGUF). Reads the *.gguf off the volume and pushes
# them — avoids a local pull (user is disk-constrained). Needs a WRITE token:
# the existing `huggingface` Modal secret is anon (Gemma downloads only), so
# this uses a separate `huggingface-write` secret exposing HF_TOKEN. Create it:
#   modal secret create huggingface-write HF_TOKEN=hf_xxx
# Defaults to a PRIVATE repo (publishing weights is hard to undo); flip with
# --private false once eyeballed.
upload_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub", "hf_transfer").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

_MODEL_CARD = """\
---
license: gemma
base_model: google/gemma-4-26b-a4b-it
tags:
  - gguf
  - visual-novel
  - gemma
  - moe
  - text-generation
pipeline_tag: text-generation
---

# Gemma 4 26B-A4B — Visual Novel (GGUF)

A QLoRA fine-tune of **`google/gemma-4-26b-a4b-it`** (26B MoE, ~4B active) for
the [Ars-Fabula](https://github.com/) anime visual-novel engine. The model
narrates slice-of-life scenes for a locked cast and drives sprites, backgrounds
and choices through a bracketed tool protocol
(`[TOOL: name key="value" choices='[...]']`).

## Training

- **Method:** QLoRA on the **MoE experts** (Path A "un-fuse"): each fused 3D
  `Gemma4TextExperts` weight is split into per-expert `nn.Linear` leaves so LoRA
  can target `experts.(gate_up|down).N` (7885 LoRA modules, 1.88% trainable),
  then merged + **re-fused** bit-exact to the canonical Gemma4 layout for GGUF.
- **Data:** 2,752 VN-protocol turns (traced real play, validity-filtered).
- **Schedule:** 2 epochs, single B200, 688 optimizer steps, final train loss 0.225.
- **Why the experts:** training the FFN experts (where phrasing lives) is what
  moved prose *quality* — attention-only and dense-FFN-only LoRA gave only a
  shallow restyle. The experts tune keeps the base model's already-low
  canned-phrase rate while raising vocabulary diversity and sharpening character
  voice (see Evaluation).

## Evaluation

This fine-tune (at its ship config, **temperature 1.1**) vs the untuned base
**`google/gemma-4-26b-a4b-it`**, on held-out VN-protocol prompts. Higher is
better for ↑ metrics, lower for ↓.

| Metric | base `gemma-4-26b-a4b-it` | **26B experts tune** |
|---|---|---|
| Protocol validity ↑ | 100% (8/8) | **100% (48/48)** |
| Slop-tell density /1k words ↓ | 2.20 | **2.11** |
| Type-token ratio (TTR) ↑ | 0.49 | **0.53** |
| Cross-scene trigram reuse ↓ | 0.019 | **0.004** |

- The base model is already strong on the protocol, so the win isn't "teaching
  the format" — it's **prose quality at no validity cost**. The tune holds 100%
  validity across **48 held-out scenes** (6 seeds) at temperature 1.1, while
  raising vocabulary diversity (TTR 0.49→0.53) and cutting verbatim cross-scene
  phrase reuse ~5× (0.019→0.004). Slop-tell density is a wash (2.20 vs 2.11) — the
  base was never slop-heavy on this list; the tune's gain is voice and variety,
  not de-cliché-ing.
- **Sampling provenance:** tune figures are mean over 6 seeds at temp 1.1
  (validity over all 48 scenes); the base was sampled once (seed 42) at temp 0.8,
  its eval default, on the same 8-scene prompt set.

### What the metrics mean

- **Protocol validity** — fraction of generated turns that pass the engine's
  own validator (`vn_validate`): well-formed `[TOOL: …]` calls, parseable
  `choices` JSON, and *cast-lock* (only the locked cast may speak/act). A hard
  well-formedness gate ("does the turn drive the UI without erroring"), not a
  taste score.
- **Slop-tell density** — count of curated "LLM-slop" phrases (stock clichés like
  *"the air hung heavy with unspoken words"*, *"a mix of X and Y"*) per **1,000
  words**, via `tools/repetition_metrics.py` against a hand-curated tell list.
  Lower = less formulaic. (The list was curated from observed *tuned*-model
  failure modes, so it may undercount base-specific clichés — read the base
  number as a floor, not a like-for-like.)
- **Type-token ratio (TTR)** — unique words ÷ total words: a vocabulary-diversity
  proxy. Higher = richer, less word-level repetition.
- **Cross-scene trigram reuse** — fraction of distinct 3-word sequences that
  recur across *different* scenes: a verbatim self-plagiarism proxy. Lower = the
  model reuses fewer canned spans from scene to scene.

### Qualitative read (48 scenes, temp 1.1, seeds 1–6)

- **Strengths:** genuinely good comedy — per-seed-varied gags with setup /
  escalation / button (i.e. composing, not memorizing, despite the low 0.225
  loss); environmental staging (shows before it tells); distinct, light character
  voices.
- **Weaknesses:** romance is the weak suit (stock, on-the-nose, little subtext);
  the "air heavy / charged with unspoken X" reflex survives the tune but clusters
  almost entirely in romance scenes; choices lean on an *open-up / deflect /
  stay-silent* triad; the occasional garbled line or first↔second-person POV slip.
- **Verdict:** read it for comedy and cast chemistry; skim the kissing scenes.

## Quants

| File | Bits | Size | Notes |
|------|------|------|-------|
| `vn26b-experts-v1-Q4_K_M.gguf` | Q4_K_M | 16.8 GB | servable default; matches the stock `gemma-4-26B-A4B-it-UD-Q4_K_M` layout |
| `vn26b-experts-v1-Q8_0.gguf`   | Q8_0   | 26.9 GB | near-lossless |

**MoE fallback (benign):** 60/658 tensors (`ffn_down` / `ffn_down_exps`, cols
704 & 2112, not ÷256) fall back `q4_K→q5_0`, `q6_K→q8_0` — so the down-projs are
*higher* precision than nominal (why Q4_K_M is 16.8 GB, not ~14 GB).

## Serving

Gemma 4 26B-A4B is a custom MoE arch; serve with the
**`atomic-llama-cpp-turboquant`** fork (the stock llama.cpp lacks the tensor
maps). The tuned model emits a reasoning channel (`<|channel>thought … <channel|>`)
that the OpenAI `/v1/chat/completions` parser mangles — hit the raw `/completion`
endpoint with the embedded chat template and strip a stray leading `<channel|>`.

**Recommended sampling:** `temperature 1.1`, `top_p 0.95`.

**Runtime** (Q4_K_M, fork `llama-server` CUDA build on a single L4, all 30 layers
offloaded `-ngl 99`): prompt eval ≈1620 tok/s, generation ≈61 tok/s.
"""


@app.function(
    image=upload_image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface-write")],
    timeout=60 * MINUTES,
)
def upload_hf(run_name: str, repo_id: str, private: bool = True,
              quants: str = "Q4_K_M,Q8_0"):
    """Push the run's quantized GGUFs + a model card to a HF repo, straight from
    the volume. Token comes from the `huggingface-write` secret (HF_TOKEN)."""
    import os as _os
    from huggingface_hub import HfApi

    token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGINGFACE_TOKEN")
    assert token, "no HF_TOKEN in the huggingface-write secret"
    api = HfApi(token=token)
    who = api.whoami()["name"]
    # if a bare repo name was passed, scope it to the authed user's namespace
    if "/" not in repo_id:
        repo_id = f"{who}/{repo_id}"
    print(f"authed as {who}; target {repo_id} (private={private})", flush=True)

    api.create_repo(repo_id, repo_type="model", private=private,
                    exist_ok=True)
    api.upload_file(path_or_fileobj=_MODEL_CARD.encode(),
                    path_in_repo="README.md", repo_id=repo_id,
                    commit_message="model card")
    print("uploaded README.md", flush=True)

    for q in [s.strip() for s in quants.split(",") if s.strip()]:
        local = f"/vol/runs/{run_name}/{run_name}-{q}.gguf"
        assert _os.path.exists(local), f"missing {local}"
        size = _os.path.getsize(local) / 1e9
        print(f"uploading {q} ({size:.1f} GB)…", flush=True)
        api.upload_file(path_or_fileobj=local,
                        path_in_repo=f"{run_name}-{q}.gguf",
                        repo_id=repo_id,
                        commit_message=f"add {q} quant")
        print(f"  done {q}", flush=True)
    url = f"https://huggingface.co/{repo_id}"
    print(f"\nDONE -> {url}", flush=True)
    return url


@app.local_entrypoint()
def main(base_model: str = "google/gemma-4-12b-it",
         data: str = "/vol/data/train.jsonl",
         run_name: str = "smoke",
         lora_r: int = 32, lora_alpha: int = 64,
         max_steps: int = -1, epochs: float = 1.0, lr: float = 1e-4,
         max_seq_len: int = 6144, grad_accum: int = 8, batch_size: int = 1,
         merge: bool = False, mode: str = "train", attn_only: bool = False,
         unfuse_experts: bool = False, temps: str = "0.8,1.1,1.4",
         seed: int = 42, types: str = "Q4_K_M,Q8_0",
         quant: str = "Q4_K_M", indices: str = "0,4,8,12",
         repo_id: str = "", private: bool = True):
    if mode == "eval-base":
        print(eval_base.remote(base_model=base_model, data_path=data,
                               run_name=run_name))
        return
    if mode == "regen":
        print(regen.remote(base_model=base_model, run_name=run_name,
                           temps=temps, unfuse_experts=unfuse_experts,
                           seed=seed))
        return
    if mode == "merge-refuse":
        print(merge_refuse.remote(base_model=base_model, run_name=run_name))
        return
    if mode == "convert-gguf":
        print(convert_gguf.remote(run_name=run_name))
        return
    if mode == "quantize":
        print(quantize.remote(run_name=run_name, types=types))
        return
    if mode == "verify-gguf":
        print(verify_gguf.remote(run_name=run_name, quant=quant,
                                 indices=indices))
        return
    if mode == "verify-chat":
        print(verify_chat.remote(run_name=run_name, quant=quant,
                                 indices=indices))
        return
    if mode == "upload-hf":
        assert repo_id, "--repo-id required (e.g. user/Gemma-26B-A4B-VisualNovel-GGUF)"
        print(upload_hf.remote(run_name=run_name, repo_id=repo_id,
                               private=private, quants=types))
        return
    if mode == "probe":
        print(probe.remote(base_model=base_model, run_name=run_name,
                           unfuse_experts=unfuse_experts))
        return
    print(train.remote(base_model=base_model, data_path=data,
                       run_name=run_name, lora_r=lora_r,
                       lora_alpha=lora_alpha, max_steps=max_steps,
                       epochs=epochs, lr=lr, max_seq_len=max_seq_len,
                       grad_accum=grad_accum, batch_size=batch_size, merge=merge,
                       attn_only=attn_only, unfuse_experts=unfuse_experts))
