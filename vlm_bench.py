#!/usr/bin/env python3
"""VLM benchmark: caption + JSON extraction + PPE detection"""

import json
import time
import base64
import argparse
import glob
import os
import httpx

PROMPT_CAPTION = """この画像を日本語で詳しく説明してください。場所、人物、物体、雰囲気などを含めてください。"""

PROMPT_JSON = """この画像から以下の情報をJSON形式で抽出してください:
- location: 場所
- event_type: イベントの種類
- subjects: 主要な被写体のリスト
- technologies: 技術やブランド名
- people_count: 人数
- atmosphere: 雰囲気

JSONのみを返してください。"""

PROMPT_PPE = """この画像に写っている作業員の安全保護具（PPE）の着用状況を分析し、以下のJSON形式で返してください:
{
  "workers_count": 人数,
  "ppe_items": [
    {
      "worker_id": 番号,
      "hard_hat": true/false,
      "safety_vest": true/false,
      "safety_glasses": true/false,
      "gloves": true/false,
      "safety_shoes": true/false,
      "other": []
    }
  ],
  "compliance_score": "高/中/低",
  "observations": "所見"
}

JSONのみを返してください。"""

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext.lstrip("."), "image/jpeg")

def call_vlm(client, api_url, model, image_path, prompt, max_tokens=1024):
    b64 = image_to_base64(image_path)
    mime = get_mime(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}
    ]

    t0 = time.time()
    resp = client.post(api_url, json={
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    latency = time.time() - t0
    data = resp.json()

    if "choices" not in data:
        return {"error": str(data)[:200], "latency": latency}

    content = data["choices"][0]["message"].get("content", "")
    timings = data.get("timings", {})

    return {
        "content": content,
        "latency": round(latency, 3),
        "tok_s": round(timings.get("predicted_per_second", 0), 1),
        "prompt_tok_s": round(timings.get("prompt_per_second", 0), 1),
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
    }

def try_parse_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        json.loads(text.strip())
        return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--image-dir", default=os.path.expanduser("~/vlm-test-images"))
    parser.add_argument("--output", default="vlm_results.json")
    args = parser.parse_args()

    client = httpx.Client(timeout=300.0)

    images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) +
                    glob.glob(os.path.join(args.image_dir, "*.png")))

    tech_images = [f for f in images if not os.path.basename(f).startswith("ppe")]
    ppe_images = [f for f in images if os.path.basename(f).startswith("ppe")]

    results = {"model": args.model, "captions": [], "json_extract": [], "ppe": []}

    # === Caption ===
    print("=== Caption Generation ===")
    for img in tech_images:
        name = os.path.basename(img)
        print(f"  {name} ...", end=" ", flush=True)
        r = call_vlm(client, args.api_url, args.model, img, PROMPT_CAPTION)
        r["image"] = name
        results["captions"].append(r)
        print(f"{r['latency']}s, {r.get('tok_s',0)} tok/s")

    # === JSON Extraction ===
    print("\n=== JSON Tag Extraction ===")
    for img in tech_images:
        name = os.path.basename(img)
        print(f"  {name} ...", end=" ", flush=True)
        r = call_vlm(client, args.api_url, args.model, img, PROMPT_JSON)
        r["image"] = name
        r["json_parse_ok"] = try_parse_json(r.get("content", ""))
        results["json_extract"].append(r)
        print(f"{r['latency']}s, parse={'✓' if r['json_parse_ok'] else '✗'}")

    # === PPE Detection ===
    print("\n=== PPE Detection ===")
    for img in ppe_images:
        name = os.path.basename(img)
        print(f"  {name} ...", end=" ", flush=True)
        r = call_vlm(client, args.api_url, args.model, img, PROMPT_PPE)
        r["image"] = name
        r["json_parse_ok"] = try_parse_json(r.get("content", ""))
        results["ppe"].append(r)
        print(f"{r['latency']}s, parse={'✓' if r['json_parse_ok'] else '✗'}")

    # === Summary ===
    cap_avg = sum(r["latency"] for r in results["captions"]) / len(results["captions"]) if results["captions"] else 0
    json_avg = sum(r["latency"] for r in results["json_extract"]) / len(results["json_extract"]) if results["json_extract"] else 0
    json_ok = sum(1 for r in results["json_extract"] if r.get("json_parse_ok")) / len(results["json_extract"]) * 100 if results["json_extract"] else 0
    ppe_avg = sum(r["latency"] for r in results["ppe"]) / len(results["ppe"]) if results["ppe"] else 0
    ppe_ok = sum(1 for r in results["ppe"] if r.get("json_parse_ok")) / len(results["ppe"]) * 100 if results["ppe"] else 0

    results["summary"] = {
        "caption_avg_latency": round(cap_avg, 2),
        "json_avg_latency": round(json_avg, 2),
        "json_parse_rate": round(json_ok, 1),
        "ppe_avg_latency": round(ppe_avg, 2),
        "ppe_parse_rate": round(ppe_ok, 1),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Caption avg: {cap_avg:.2f}s ({len(results['captions'])} images)")
    print(f"JSON extract avg: {json_avg:.2f}s (parse rate: {json_ok:.0f}%)")
    print(f"PPE detect avg: {ppe_avg:.2f}s (parse rate: {ppe_ok:.0f}%)")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
