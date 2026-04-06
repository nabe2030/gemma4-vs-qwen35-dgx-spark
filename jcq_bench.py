#!/usr/bin/env python3
"""JCommonsenseQA v1.1 benchmark for OpenAI-compatible API (llama-server)"""

import json
import time
import argparse
import httpx
from datasets import load_dataset

FEW_SHOT_EXAMPLES = [
    {
        "question": "つまらない事についてああでもないこうでもないと関係ない人たちが関わる事をなんという？",
        "choices": ["野次馬", "傍観者", "見物人", "通行人", "関係者"],
        "answer": 0
    },
    {
        "question": "カードローンは何のためにする？",
        "choices": ["自動車", "お金を借りる", "旅行", "貯金", "買い物"],
        "answer": 1
    },
    {
        "question": "木材を切断するために使用される工具は？",
        "choices": ["ハンマー", "ドライバー", "ペンチ", "のこぎり", "スパナ"],
        "answer": 3
    },
]

def build_prompt(question, choices):
    lines = [f"質問: {question}"]
    for i, c in enumerate(choices):
        lines.append(f"{i}: {c}")
    lines.append("回答（番号のみ）:")
    return "\n".join(lines)

def build_messages(question, choices):
    messages = [
        {"role": "system", "content": "あなたは日本語の常識問題に答えるアシスタントです。選択肢の番号のみを回答してください。"}
    ]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": build_prompt(ex["question"], ex["choices"])})
        messages.append({"role": "assistant", "content": str(ex["answer"])})
    messages.append({"role": "user", "content": build_prompt(question, choices)})
    return messages

def extract_answer(text):
    if not text:
        return -1
    text = text.strip()
    for ch in text:
        if ch in "01234":
            return int(ch)
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", default="jcq_results.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ds = load_dataset("leemeng/jcommonsenseqa-v1.1", split="validation")
    items = list(ds)
    if args.limit > 0:
        items = items[:args.limit]

    results = []
    correct = 0
    total_latency = 0.0
    total_tps_list = []

    client = httpx.Client(timeout=300.0)

    for i, item in enumerate(items):
        choices = [item[f"choice{j}"] for j in range(5)]
        messages = build_messages(item["question"], choices)

        t0 = time.time()
        try:
            resp = client.post(args.api_url, json={
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            })
            data = resp.json()

            if "choices" not in data:
                print(f"[{i+1}] API error: {json.dumps(data, ensure_ascii=False)[:200]}")
                results.append({"q_id": item["q_id"], "question": item["question"],
                                "gold": item["label"], "pred": -1, "correct": False,
                                "response": str(data), "latency": 0})
                continue

            msg = data["choices"][0]["message"]
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            finish = data["choices"][0].get("finish_reason", "")

            # If thinking exhausted tokens and content empty, retry
            if not content.strip() and finish == "length":
                resp2 = client.post(args.api_url, json={
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.0,
                })
                data = resp2.json()
                if "choices" in data:
                    msg = data["choices"][0]["message"]
                    content = msg.get("content", "")
                    reasoning = msg.get("reasoning_content", "")

            # Extract tok/s from timings
            timings = data.get("timings", {})
            pred_tps = timings.get("predicted_per_second", 0)
            prompt_tps = timings.get("prompt_per_second", 0)
            if pred_tps > 0:
                total_tps_list.append(pred_tps)

        except Exception as e:
            print(f"[{i+1}] Error: {e}")
            results.append({"q_id": item["q_id"], "question": item["question"],
                            "gold": item["label"], "pred": -1, "correct": False,
                            "response": str(e), "latency": 0})
            continue

        latency = time.time() - t0
        total_latency += latency

        pred = extract_answer(content)
        gold = item["label"]
        is_correct = pred == gold

        if is_correct:
            correct += 1

        results.append({
            "q_id": item["q_id"],
            "question": item["question"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response": content,
            "reasoning": reasoning[:200] if reasoning else "",
            "latency": round(latency, 3),
            "tok_s": round(pred_tps, 1),
        })

        acc = correct / (i + 1) * 100
        think_flag = "T" if reasoning else ""
        print(f"[{i+1}/{len(items)}] acc={acc:.1f}% lat={latency:.2f}s tps={pred_tps:.1f} pred={pred} gold={gold} {'✓' if is_correct else '✗'} {think_flag}")

    avg_latency = total_latency / len(items) if items else 0
    avg_tps = sum(total_tps_list) / len(total_tps_list) if total_tps_list else 0

    summary = {
        "model": args.model,
        "total": len(items),
        "correct": correct,
        "accuracy": round(correct / len(items) * 100, 2),
        "avg_latency": round(avg_latency, 3),
        "avg_tok_s": round(avg_tps, 1),
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Results ===")
    print(f"Model: {args.model}")
    print(f"Accuracy: {correct}/{len(items)} ({summary['accuracy']}%)")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"Avg tok/s: {avg_tps:.1f}")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
