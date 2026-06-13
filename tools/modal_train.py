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


@app.function(
    image=image,
    gpu=os.environ.get("ARS_FABULA_TRAIN_GPU", "L40S"),
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=6 * 60 * MINUTES,
)
def train(base_model: str, data_path: str, run_name: str,
          lora_r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.05,
          max_steps: int = -1, epochs: float = 1.0, lr: float = 1e-4,
          max_seq_len: int = 6144, batch_size: int = 1,
          grad_accum: int = 8, eval_holdout: int = 8,
          merge: bool = False):
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq)
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
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
    module_names = [n for n, _ in model.named_modules()]
    if any("language_model" in n for n in module_names):
        # unified multimodal arch: adapt ONLY the language tower, not the
        # vision/audio encoders
        targets = r".*language_model.*\.({})$".format("|".join(proj_names))
    else:
        targets = proj_names
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
        save_strategy="no",
        gradient_checkpointing=True,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=SFTDataset(trainset),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()
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


@app.local_entrypoint()
def main(base_model: str = "google/gemma-4-12b-it",
         data: str = "/vol/data/train.jsonl",
         run_name: str = "smoke",
         lora_r: int = 32, lora_alpha: int = 64,
         max_steps: int = -1, epochs: float = 1.0, lr: float = 1e-4,
         max_seq_len: int = 6144, grad_accum: int = 8,
         merge: bool = False, mode: str = "train"):
    if mode == "eval-base":
        print(eval_base.remote(base_model=base_model, data_path=data,
                               run_name=run_name))
        return
    print(train.remote(base_model=base_model, data_path=data,
                       run_name=run_name, lora_r=lora_r,
                       lora_alpha=lora_alpha, max_steps=max_steps,
                       epochs=epochs, lr=lr, max_seq_len=max_seq_len,
                       grad_accum=grad_accum, merge=merge))
