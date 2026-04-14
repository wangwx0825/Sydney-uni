#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import math
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

OUTPUT_PREFIX = "composite_confidence_100q_10runs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="train.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def get_torch_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_jsonl(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_first_key(d, candidates):
    for key in candidates:
        if key in d and d[key] is not None:
            return d[key]
    return None


def get_question(row):
    q = pick_first_key(
        row,
        ["question", "input", "prompt", "instruction", "query", "problem"]
    )
    if q is None:
        raise KeyError(f"Cannot find question field. Keys: {list(row.keys())}")
    if isinstance(q, list):
        q = "\n".join(str(x) for x in q)
    return str(q).strip()


def get_gold_answer(row):
    a = pick_first_key(
        row,
        ["answer", "output", "target", "label", "gold", "reference_answer"]
    )
    if a is None:
        raise KeyError(f"Cannot find answer field. Keys: {list(row.keys())}")
    return str(a).strip()


def extract_gsm8k_final(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    return None


def normalize_text(s):
    s = str(s).strip()
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def normalize_answer(s):
    s = str(s).strip()

    gsm = extract_gsm8k_final(s)
    if gsm:
        s = gsm

    boxed = re.findall(r"\\boxed\{([^}]*)\}", s)
    if boxed:
        s = boxed[-1].strip()

    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if nums:
        last_num = nums[-1]
        try:
            val = float(last_num)
            if abs(val - int(val)) < 1e-9:
                return str(int(val))
            return str(val)
        except Exception:
            pass

    s = normalize_text(s)
    s = re.sub(r"^[\s:;\-]*", "", s)
    s = re.sub(r"[\s\.\!\?]+$", "", s)
    return s


def answers_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def safe_json_load(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            try:
                return ast.literal_eval(blob)
            except Exception:
                return None
    return None


def build_messages(question):
    system = (
        "Solve the problem carefully.\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{"reasoning": "...", "final_answer": "..."}\n'
        "Do not use markdown fences. Do not add extra text outside JSON."
    )
    user = (
        f"Question:\n{question}\n\n"
        "Provide concise reasoning and then a final answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def prepare_model(args):
    dtype = get_torch_dtype(args.dtype)
    quant_config = None

    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        quantization_config=quant_config,
    )
    model.eval()
    return tokenizer, model


def parse_model_output(raw_text):
    data = safe_json_load(raw_text)
    if not isinstance(data, dict):
        return "", raw_text.strip()

    reasoning = str(data.get("reasoning", "")).strip()
    final_answer = str(data.get("final_answer", "")).strip()
    return reasoning, final_answer


def compute_token_metrics(generation_output, prompt_len):
    sequences = generation_output.sequences
    generated_ids = sequences[0, prompt_len:]
    scores = generation_output.scores

    if len(scores) == 0 or generated_ids.numel() == 0:
        return {
            "entropy": np.nan,
            "mean_logprob": np.nan,
            "min_logprob": np.nan,
            "top1_top2_margin": np.nan,
            "token_prob_std": np.nan,
        }

    entropies = []
    chosen_probs = []
    margins = []

    steps = min(len(scores), generated_ids.numel())

    for step_idx in range(steps):
        logits = scores[step_idx][0].float()
        probs = torch.softmax(logits, dim=-1)

        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
        entropies.append(entropy)

        chosen_id = generated_ids[step_idx].item()
        chosen_prob = probs[chosen_id].item()
        chosen_probs.append(chosen_prob)

        top2 = torch.topk(probs, k=2).values
        margin = (top2[0] - top2[1]).item()
        margins.append(margin)

    chosen_probs_np = np.array(chosen_probs, dtype=np.float64)
    chosen_logprobs_np = np.log(np.clip(chosen_probs_np, 1e-12, 1.0))

    return {
        "entropy": float(np.mean(entropies)),
        "mean_logprob": float(np.mean(chosen_logprobs_np)),
        "min_logprob": float(np.min(chosen_logprobs_np)),
        "top1_top2_margin": float(np.mean(margins)),
        "token_prob_std": float(np.std(chosen_probs_np)),
    }


@torch.no_grad()
def sample_once(tokenizer, model, question, max_new_tokens, temperature, top_p):
    messages = build_messages(question)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output.sequences[0, prompt_len:], skip_special_tokens=True)
    reasoning, pred_answer = parse_model_output(decoded)
    token_metrics = compute_token_metrics(output, prompt_len)

    return {
        "raw_output": decoded,
        "reasoning": reasoning,
        "pred_answer": pred_answer,
        **token_metrics,
    }


def add_agreement_rate(df):
    pieces = []

    for question_id, g in df.groupby("question_id", sort=True):
        preds = [normalize_answer(x) for x in g["pred_answer"].tolist()]
        counts = Counter(preds)
        majority_answer, majority_count = counts.most_common(1)[0]
        agreement_rate = majority_count / len(preds)

        g = g.copy()
        g["majority_answer"] = majority_answer
        g["agreement_rate"] = agreement_rate
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


def minmax_scale(series, higher_is_better=True):
    s = series.astype(float).copy()
    valid = s.dropna()

    if len(valid) == 0:
        return pd.Series(np.nan, index=s.index)

    vmin = valid.min()
    vmax = valid.max()

    if abs(vmax - vmin) < 1e-12:
        scaled = pd.Series(0.5, index=s.index)
    else:
        scaled = (s - vmin) / (vmax - vmin)

    if not higher_is_better:
        scaled = 1.0 - scaled

    return scaled.clip(0.0, 1.0)


def build_composite_confidence(df):
    out = df.copy()

    # 方向说明：
    # agreement_rate: 越高越好
    # entropy: 越低越好
    # mean_logprob: 越高越好（越不负越好）
    # min_logprob: 越高越好
    # top1_top2_margin: 越高越好
    # token_prob_std: 这里按越低越稳处理

    out["agreement_rate_score"] = minmax_scale(out["agreement_rate"], higher_is_better=True)
    out["entropy_score"] = minmax_scale(out["entropy"], higher_is_better=False)
    out["mean_logprob_score"] = minmax_scale(out["mean_logprob"], higher_is_better=True)
    out["min_logprob_score"] = minmax_scale(out["min_logprob"], higher_is_better=True)
    out["top1_top2_margin_score"] = minmax_scale(out["top1_top2_margin"], higher_is_better=True)
    out["token_prob_std_score"] = minmax_scale(out["token_prob_std"], higher_is_better=False)

    score_cols = [
        "agreement_rate_score",
        "entropy_score",
        "mean_logprob_score",
        "min_logprob_score",
        "top1_top2_margin_score",
        "token_prob_std_score",
    ]

    out["composite_score"] = out[score_cols].mean(axis=1)
    out["confidence"] = out["composite_score"].clip(0.0, 1.0)

    return out


def build_question_summary(df):
    summary = (
        df.groupby("question_id", as_index=False)
        .agg({
            "question": "first",
            "gold_answer": "first",
            "agreement_rate": "first",
            "confidence": "mean",
            "accuracy": "mean",
            "entropy": "mean",
            "mean_logprob": "mean",
            "min_logprob": "mean",
            "top1_top2_margin": "mean",
            "token_prob_std": "mean",
        })
        .rename(columns={
            "confidence": "avg_confidence",
            "accuracy": "accuracy_rate",
            "entropy": "avg_entropy",
            "mean_logprob": "avg_mean_logprob",
            "min_logprob": "avg_min_logprob",
            "top1_top2_margin": "avg_top1_top2_margin",
            "token_prob_std": "avg_token_prob_std",
        })
    )
    return summary


def plot_confidence_accuracy(summary_df, prefix):
    plt.figure(figsize=(7, 7))
    plt.scatter(summary_df["avg_confidence"], summary_df["accuracy_rate"], alpha=0.7, s=28)

    x = np.linspace(0, 1, 100)
    plt.plot(x, x, linestyle="--")

    plt.xlabel("average confidence")
    plt.ylabel("accuracy rate")
    plt.title("confidence vs accuracy (100 questions)")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(f"{prefix}_confidence_accuracy_scatter.png", dpi=220)
    plt.close()


def plot_binned_confidence_accuracy(summary_df, prefix):
    clean = summary_df[["avg_confidence", "accuracy_rate"]].dropna().copy()
    if clean.empty:
        return

    try:
        q = min(10, clean["avg_confidence"].nunique())
        if q < 2:
            return
        clean["bin"] = pd.qcut(clean["avg_confidence"], q=q, duplicates="drop")
    except ValueError:
        return

    binned = clean.groupby("bin", observed=False).agg(
        confidence_mean=("avg_confidence", "mean"),
        accuracy_mean=("accuracy_rate", "mean"),
        count=("accuracy_rate", "size"),
    ).reset_index(drop=True)

    plt.figure(figsize=(7, 5))
    plt.plot(binned["confidence_mean"], binned["accuracy_mean"], marker="o")
    plt.xlabel("average confidence")
    plt.ylabel("average accuracy")
    plt.title("confidence vs accuracy (binned)")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(f"{prefix}_confidence_accuracy_binned.png", dpi=220)
    plt.close()

    binned.to_csv(f"{prefix}_confidence_accuracy_bin_stats.csv", index=False)


def main():
    args = parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] Loading dataset: {args.dataset}")
    rows = load_jsonl(args.dataset, args.limit)
    print(f"[INFO] Loaded {len(rows)} rows")

    print(f"[INFO] Loading model: {args.model}")
    tokenizer, model = prepare_model(args)

    all_records = []

    for qid, row in enumerate(rows):
        question = get_question(row)
        gold_answer = get_gold_answer(row)

        print(f"[INFO] Question {qid + 1}/{len(rows)}")

        for run_idx in range(args.repeats):
            try:
                sample = sample_once(
                    tokenizer=tokenizer,
                    model=model,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                accuracy = answers_match(sample["pred_answer"], gold_answer)

                record = {
                    "question_id": qid,
                    "run_id": run_idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "reasoning": sample["reasoning"],
                    "pred_answer": sample["pred_answer"],
                    "raw_output": sample["raw_output"],
                    "entropy": sample["entropy"],
                    "mean_logprob": sample["mean_logprob"],
                    "min_logprob": sample["min_logprob"],
                    "top1_top2_margin": sample["top1_top2_margin"],
                    "token_prob_std": sample["token_prob_std"],
                    "accuracy": accuracy,
                }
            except Exception as e:
                record = {
                    "question_id": qid,
                    "run_id": run_idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "reasoning": "",
                    "pred_answer": "",
                    "raw_output": f"[ERROR] {repr(e)}",
                    "entropy": np.nan,
                    "mean_logprob": np.nan,
                    "min_logprob": np.nan,
                    "top1_top2_margin": np.nan,
                    "token_prob_std": np.nan,
                    "accuracy": 0,
                }

            all_records.append(record)

    runs_df = pd.DataFrame(all_records)
    runs_df = add_agreement_rate(runs_df)
    runs_df = build_composite_confidence(runs_df)

    runs_df.to_csv(f"{OUTPUT_PREFIX}_runs.csv", index=False)
    print(f"[INFO] Saved {OUTPUT_PREFIX}_runs.csv")

    question_summary_df = build_question_summary(runs_df)
    question_summary_df.to_csv(f"{OUTPUT_PREFIX}_question_summary.csv", index=False)
    print(f"[INFO] Saved {OUTPUT_PREFIX}_question_summary.csv")

    plot_confidence_accuracy(question_summary_df, OUTPUT_PREFIX)
    print(f"[INFO] Saved {OUTPUT_PREFIX}_confidence_accuracy_scatter.png")

    plot_binned_confidence_accuracy(question_summary_df, OUTPUT_PREFIX)
    print(f"[INFO] Saved {OUTPUT_PREFIX}_confidence_accuracy_binned.png")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
