"""QLoRA fine-tune of Gemma-3-4B-it on the distillation set (Claude teacher labels). $0 API.

QLoRA in one paragraph: the base model's weights are frozen and stored in 4-bit (bitsandbytes NF4),
and small trainable low-rank matrices (LoRA, rank r) are added next to the attention and MLP
projections. Only those (~1% of parameters) get gradients and optimizer state, which is why a 4B
model trains on an 8 GB laptop GPU. Gradient checkpointing trades compute for activation memory.

  uv run python scripts/train_qlora.py [--max-steps N] [--epochs 2]

Details that matter:
- Same chat template and the exact judge prompt as serving (system + user), target = teacher JSON.
- Loss only on the assistant answer: prompt tokens are masked with -100.
- LoRA only on the LANGUAGE model (this checkpoint also has a vision tower we don't train).
- 5% of examples held out by contract (not by row) to watch validation loss; evals use the test split later.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "hf" / "gemma-3-4b-it-bnb-4bit"
DATA = ROOT / "data" / "train" / "distill_haiku_rag10.jsonl"
OUT = ROOT / "models" / "lora" / "gemma-3-4b-clausecheck"
MAX_LEN = 3072
TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


class SFTData(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], tok, drop_over: int = MAX_LEN) -> None:
        self.items, self.dropped = [], 0
        for r in rows:
            prompt = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
            p_text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            full_text = tok.apply_chat_template(prompt + [{"role": "assistant", "content": r["assistant"]}],
                                                tokenize=False)
            p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
            ids = tok(full_text, add_special_tokens=False)["input_ids"]
            if len(ids) > drop_over:
                self.dropped += 1
                continue
            labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
            if any(l != -100 for l in labels):
                self.items.append((ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    # Gemma 3 (vision-language) wants token_type_ids in training: 0 = text token
    return {"input_ids": ids, "labels": lab, "attention_mask": att, "token_type_ids": torch.zeros_like(ids)}


def shared_gpu_gb(pid: int) -> float:
    """Windows only: GPU memory this process has spilled into system RAM ("shared usage").

    When an allocation does not fit in the card's VRAM, the Windows driver silently backs it with
    system RAM instead of raising out-of-memory, and training runs several times slower. Any value
    well above ~0.3 GB here means that is happening.
    """
    import subprocess

    ps = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage').CounterSamples"
          " | Measure-Object CookedValue -Sum | ForEach-Object Sum")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip() or 0) / 1e9
    except (ValueError, subprocess.SubprocessError):
        return float("nan")


class StepMonitor(TrainerCallback):
    """Prints seconds per example, peak VRAM, and RAM spill after every optimizer step."""

    def __init__(self, examples_per_step: int) -> None:
        self.eps, self.t = examples_per_step, None

    def on_step_begin(self, args, state, control, **kw):
        self.t = time.time()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.empty_cache()  # hand back cached-but-unused blocks so the reservation can't creep upward
        dt = time.time() - self.t
        print(f"step {state.global_step}/{state.max_steps}: {dt / self.eps:.1f} s/example, "
              f"peak VRAM {torch.cuda.max_memory_reserved() / 1e9:.2f} GB, "
              f"spill to RAM {shared_gpu_gb(os.getpid()):.2f} GB", flush=True)


class AnswerOnlyTrainer(Trainer):
    """Compute logits only for the answer span.

    Gemma 3's vocabulary has 262k tokens: full logits for a 2.5k-token example are ~2.6 GB, which
    ran the 8 GB GPU out of memory. Prompt positions have label -100 anyway, so we ask the model for
    the last k positions only (`logits_to_keep`) and compute the same cross-entropy on them.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        first = int((labels != -100).float().argmax(dim=1).min())  # first answer token in the batch
        k = labels.shape[1] - first + 1  # +1: the position before the first answer token predicts it
        out = model(**inputs, logits_to_keep=k)
        logits = out.logits[:, :-1, :].float()  # positions [L-k, L-2] predict tokens [L-k+1, L-1]
        target = labels[:, -(k - 1):]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                                 ignore_index=-100, reduction="sum")
        n = num_items_in_batch if num_items_in_batch is not None else (target != -100).sum()
        loss = loss / n
        return (loss, out) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1)  # measured ~4 s/example on the laptop: 1 epoch ≈ 1.8 h
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--drop-over", type=int, default=MAX_LEN,
                    help="drop (never truncate) examples longer than this many tokens: truncation cuts off the answer")
    ap.add_argument("--probe", type=int, default=0,
                    help="worst-case memory test: train a few steps on only the N longest examples, save nothing")
    ap.add_argument("--vram-cap", type=float, default=0.88, help="max fraction of GPU memory PyTorch may reserve")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in OUT/checkpoints")
    ap.add_argument("--init-adapter", default="", help="start from this trained adapter instead of a new one")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=42, help="data-order seed (use a new one for a second epoch)")
    ap.add_argument("--extra-data", default="", help="additional JSONL training file (same row format)")
    ap.add_argument("--replay", type=int, default=0,
                    help="with --extra-data: train on the extra rows + this many sampled original rows only")
    a = ap.parse_args()
    out_dir = Path(a.out)

    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    contracts = sorted({r["key"].split("::")[0] for r in rows})
    random.Random(0).shuffle(contracts)
    val_c = set(contracts[: max(1, len(contracts) // 20)])
    train_rows = [r for r in rows if r["key"].split("::")[0] not in val_c]
    val_rows = [r for r in rows if r["key"].split("::")[0] in val_c]
    if a.extra_data:  # e.g. named-party examples; same held-out contracts stay in validation
        extra = [json.loads(l) for l in Path(a.extra_data).read_text(encoding="utf-8").splitlines() if l.strip()]
        new_train = [r for r in extra if r["key"].split("::")[0] not in val_c]
        val_rows += [r for r in extra if r["key"].split("::")[0] in val_c]
        if a.replay:  # mix in old examples so the adapter doesn't drift away from what it already does well
            train_rows = random.Random(a.seed).sample(train_rows, min(a.replay, len(train_rows)))
        train_rows = new_train + train_rows
        random.Random(a.seed).shuffle(train_rows)
        print(f"extra data: {len(new_train)} train examples from {a.extra_data}; replay {len(train_rows) - len(new_train)}")

    tok = AutoTokenizer.from_pretrained(BASE)
    train_ds, val_ds = SFTData(train_rows, tok, a.drop_over), SFTData(val_rows, tok, a.drop_over)
    print(f"examples: train {len(train_ds)}, val {len(val_ds)} (val = {len(val_c)} held-out contracts); "
          f"dropped over {a.drop_over} tokens: {train_ds.dropped + val_ds.dropped}", flush=True)
    if a.probe:
        train_ds.items = sorted(train_ds.items, key=lambda x: -len(x[0]))[: a.probe]
        a.max_steps = max(1, a.probe // 8)
        print(f"PROBE: {a.probe} longest examples ({len(train_ds.items[-1][0])}-{len(train_ds.items[0][0])} tokens), "
              f"{a.max_steps} steps", flush=True)

    # Cap PyTorch's share of the card. Measured failure without it: the caching allocator kept reserving
    # more as sequence lengths varied (7.0 → 7.9 → 8.8 → 9.4 GB on an 8 GB card) and Windows silently backed
    # the excess with system RAM: 2.9 → 21 s/example. With a cap, the allocator frees its cache and retries
    # instead of growing; if the real working set doesn't fit, we get a clear out-of-memory error, not a 7x slowdown.
    torch.cuda.set_per_process_memory_fraction(a.vram_cap)
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
    # NOT prepare_model_for_kbit_training(): it upcasts non-quantized weights to fp32, and Gemma's 262k-token
    # embedding table (~671M params) goes 1.3 → 2.7 GB. Measured: 4.71 GB after load vs ~3.3 GB without,
    # which pushed training past 8 GB into the Windows system-RAM spill. We enable what we need directly.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.init_adapter:  # continue training an existing adapter (e.g. epoch 2 with a fresh, lower-LR schedule)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, seed=a.seed, lr_scheduler_type="cosine",
        warmup_steps=10, bf16=True,  # transformers 5 dropped warmup_ratio
        logging_steps=5, eval_strategy="steps", eval_steps=50, disable_tqdm=True,
        per_device_eval_batch_size=1, save_strategy="no" if a.probe else "steps", save_steps=40, save_total_limit=1,
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
    )
    trainer = AnswerOnlyTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id),
                      callbacks=[StepMonitor(args.gradient_accumulation_steps)])
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if a.resume else None)
    mins = (time.time() - t0) / 60
    if a.probe:
        print(f"PROBE done in {mins:.1f} min, peak VRAM reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    hist = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    (out_dir / "train_log.json").write_text(json.dumps({"minutes": mins, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                                                    "history": hist}, indent=1), encoding="utf-8")
    print(f"done in {mins:.1f} min, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, adapter → {out_dir}")


if __name__ == "__main__":
    main()
