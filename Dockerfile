# Ars Fabula VN — Hugging Face Docker Space (paid GPU: Nvidia T4 medium,
# 16GB VRAM / 30GB RAM / 8 vCPU).
#
# Runs the FULL stack, same architecture as the dev machine:
#   llama-server (stock upstream, Gemma 4 26B-A4B Q4_K_M + MTP drafter,
#                 --cpu-moe: experts in RAM ~16GB, GPU ~5.5GB)
#   ComfyUI      (anima Q5 GGUF UNET + Qwen encoder/VAE + Impact face gate,
#                 ~3-4GB VRAM when hot)
#   Gradio app   (ARS_FABULA_BACKEND=server, ARS_FABULA_BG=comfyui)
#
# Both stacks fit the T4 together (~9-10GB peak); the env-driven VRAM swap
# (ARS_FABULA_LLM_STOP_CMD/START_CMD) stays wired as the OOM escape hatch.
# All model downloads are from UNGATED repos — no build secrets needed.

# ── Stage 1: build llama-server (static, SM75 for T4) ───────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS llama-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl ca-certificates python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir cmake ninja

# Pinned to the master commit validated locally (MTP merged 2026-06-07).
ARG LLAMA_COMMIT=1593d56
RUN git clone https://github.com/ggml-org/llama.cpp /llama.cpp \
    && cd /llama.cpp && git checkout ${LLAMA_COMMIT}
RUN cd /llama.cpp \
    && cmake -B build -G Ninja \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && ninja -C build llama-server

# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /app

COPY --from=llama-build /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# ── ComfyUI + custom nodes (pinned to the validated local install) ──────
ARG COMFY_COMMIT=b6332446
ARG IMPACT_SUBPACK_COMMIT=50c7b71
RUN git clone https://github.com/comfyanonymous/ComfyUI /app/comfy \
    && cd /app/comfy && git checkout ${COMFY_COMMIT} \
    && git clone https://github.com/city96/ComfyUI-GGUF /app/comfy/custom_nodes/ComfyUI-GGUF \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack /app/comfy/custom_nodes/comfyui-impact-pack \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack /app/comfy/custom_nodes/ComfyUI-Impact-Subpack \
    && cd /app/comfy/custom_nodes/ComfyUI-Impact-Subpack && git checkout ${IMPACT_SUBPACK_COMMIT}

RUN pip3 install --no-cache-dir -r /app/comfy/requirements.txt \
    && pip3 install --no-cache-dir -r /app/comfy/custom_nodes/ComfyUI-GGUF/requirements.txt \
    && pip3 install --no-cache-dir -r /app/comfy/custom_nodes/comfyui-impact-pack/requirements.txt \
    && pip3 install --no-cache-dir -r /app/comfy/custom_nodes/ComfyUI-Impact-Subpack/requirements.txt \
    && pip3 install --no-cache-dir "rembg[cpu]" onnxruntime

# ── App dependencies ─────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir gradio==6.16.0 requests pillow numpy \
        "huggingface_hub[hf_transfer]"
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ── Models (~20GB, baked into image layers so restarts don't re-download).
#    Every repo here is public and ungated. ──────────────────────────────
RUN hf download unsloth/gemma-4-26B-A4B-it-GGUF \
        gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --local-dir /app/models
RUN hf download unsloth/gemma-4-26B-A4B-it-GGUF \
        MTP/gemma-4-26B-A4B-it-Q8_0-MTP.gguf --local-dir /app/models
RUN hf download Abiray/Anima-base-v1.0-GGUF anima-base-v1.0-Q5_K_M.gguf \
        --local-dir /app/comfy/models/unet
RUN hf download circlestone-labs/Anima \
        split_files/text_encoders/qwen_3_06b_base.safetensors \
        --local-dir /tmp/anima \
    && mv /tmp/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
          /app/comfy/models/text_encoders/ \
    && hf download circlestone-labs/Anima \
        split_files/vae/qwen_image_vae.safetensors --local-dir /tmp/anima \
    && mv /tmp/anima/split_files/vae/qwen_image_vae.safetensors \
          /app/comfy/models/vae/ && rm -rf /tmp/anima
RUN mkdir -p /app/comfy/models/sams /app/comfy/models/ultralytics/bbox \
    && curl -L -o /app/comfy/models/sams/sam_vit_b_01ec64.pth \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    && hf download Bingsu/adetailer face_yolov8n.pt \
        --local-dir /app/comfy/models/ultralytics/bbox

# ── App code ─────────────────────────────────────────────────────────────
COPY . /app/vn
RUN chown -R user:user /app

# ── Runtime wiring (all consumed by the existing env seams) ─────────────
ENV ARS_FABULA_BACKEND=server \
    ARS_FABULA_BASE_URL=http://127.0.0.1:8110/v1 \
    ARS_FABULA_COMFY=http://127.0.0.1:8188 \
    ARS_FABULA_BG=comfyui \
    ARS_FABULA_LLM_MANAGE=cmd \
    ARS_FABULA_LLM_STOP_CMD="pkill -x llama-server" \
    ARS_FABULA_LLM_START_CMD="bash /app/vn/tools/space_llm_start.sh" \
    ARS_FABULA_LLAMA_BIN=/usr/local/bin/llama-server \
    ARS_FABULA_LLAMA_MODEL=/app/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
    ARS_FABULA_LLAMA_DRAFTER=/app/models/MTP/gemma-4-26B-A4B-it-Q8_0-MTP.gguf \
    ARS_FABULA_TRACE=0 \
    GRADIO_SERVER_PORT=7860

USER user
WORKDIR /app/vn
EXPOSE 7860
CMD ["bash", "tools/start_space.sh"]
