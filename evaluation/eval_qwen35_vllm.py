# ExtremeWhenBench
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Qwen3.5-9B evaluation on the hour-long benchmark.

Video-mode inference via vLLM: each (video, question) is sent as
`{video_url file://..., question}` and the model returns "happens in
START - END" which we parse and compare to GT via IoU.

Thinking is disabled (`enable_thinking=False`) — this is the no-think
video-mode reference run reported in the paper.

Parse failures and API errors count as IoU = 0 (strict convention).

Example:
    # 1) Serve Qwen3.5-9B with vLLM (in another shell)
    vllm serve Qwen/Qwen3-VL-A14B-Instruct --port 8000 --served-model-name qwen3.5-9b

    # 2) Run the evaluation (questions are loaded from HF by default;
    #    pass a local JSON via --bench to override)
    python eval_qwen35_vllm.py \\
        --video-dir ./videos \\
        --num-frames 768 \\
        --out qwen35_f768.jsonl
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path

from openai import AsyncOpenAI


PROMPT_TEMPLATE = (
    "[Video Temporal Grounding]\n"
    "You are watching a {dur:.0f}-second long video.\n"
    "Please find the moment described by the following question, "
    "determining its starting and ending times. The format should be: "
    "'The event happens in the start time - end time'. "
    "For example, The event 'person turn a light on' happens in the 24.3 - 30.4 seconds. "
    'Now I will give you the question: "{question}"\n'
    "Please return its start time and end time."
)

TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-~–]\s*(\d+(?:\.\d+)?)")


def parse_time(text):
    if not text:
        return None
    text_l = text.lower()
    candidates = [s for s in re.split(r"[!?\n]", text_l)
                  if "happen" in s or "start" in s or "end" in s]
    candidates = candidates or [text_l]
    for s in candidates:
        m = TIME_RE.search(s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return [a, b] if b >= a else [b, a]
    m = TIME_RE.search(text_l)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return [a, b] if b >= a else [b, a]
    return None


def iou(pred, gt):
    if pred is None:
        return 0.0
    ps, pe = pred
    gs, ge = float(gt[0]), float(gt[1])
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def get_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        timeout=30,
    ).decode().strip()
    return float(out)


async def run_one(client, q, video_dir, model, num_frames, sem):
    async with sem:
        qid, vid, question, gt = q["qid"], q["video_id"], q["question"], q["correct_interval"]
        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            return {"qid": qid, "video_id": vid, "pred": None, "iou": 0.0,
                    "error": "video not found"}
        try:
            duration = get_duration(video_path)
        except Exception as e:
            return {"qid": qid, "video_id": vid, "pred": None, "iou": 0.0,
                    "error": f"ffprobe: {str(e)[:100]}"}
        prompt = PROMPT_TEMPLATE.format(dur=duration, question=question)
        msgs = [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
            {"type": "text", "text": prompt},
        ]}]
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=0.0,
                top_p=0.8,
                max_tokens=128,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "media_io_kwargs": {"video": {"num_frames": num_frames}},
                },
                timeout=600,
            )
            text = resp.choices[0].message.content or ""
            err = None
        except Exception as e:
            text, err = "", str(e)[:300]
        pred = parse_time(text)
        return {"qid": qid, "video_id": vid, "question": question, "gt": gt,
                "duration": duration if not err else None,
                "pred": pred, "iou": iou(pred, gt),
                "response": text[:500], "elapsed_s": time.time() - t0, "error": err}


def load_questions(bench):
    """Load questions from either a local JSON path or an HF dataset name."""
    if os.path.exists(bench):
        data = json.load(open(bench))
        return data["questions"] if isinstance(data, dict) and "questions" in data else data
    from datasets import load_dataset
    ds = load_dataset(bench, split="test")
    return [dict(row) for row in ds]


async def main_async(args):
    questions = load_questions(args.bench)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Total questions: {len(questions)}", flush=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=900)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [run_one(client, q, args.video_dir, args.model, args.num_frames, sem)
             for q in questions]
    rows, started = [], time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fut in asyncio.as_completed(tasks):
        r = await fut
        rows.append(r)
        if len(rows) % 20 == 0 or len(rows) == len(tasks):
            ious = [x["iou"] for x in rows]
            miou = sum(ious) / len(rows)
            elapsed = time.time() - started
            rate = len(rows) / elapsed
            eta = (len(tasks) - len(rows)) / rate / 60 if rate else 0
            print(f"  {len(rows)}/{len(tasks)} mIoU={miou:.3f} "
                  f"({rate:.2f} q/s, ETA {eta:.1f}m)", flush=True)
            with out_path.open("w") as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with out_path.open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    ious = [x["iou"] for x in rows]; n = len(rows)
    summary = {
        "n": n, "num_frames": args.num_frames,
        "mIoU": sum(ious) / n,
        "R03": sum(1 for i in ious if i >= 0.3) / n,
        "R05": sum(1 for i in ious if i >= 0.5) / n,
        "R07": sum(1 for i in ious if i >= 0.7) / n,
        "parse_fail": sum(1 for r in rows if r["pred"] is None) / n,
        "err": sum(1 for r in rows if r.get("error")) / n,
    }
    Path(args.out.replace(".jsonl", "_summary.json")).write_text(
        json.dumps(summary, indent=2))
    print(f"\nFinal: {json.dumps(summary, indent=2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="min1321/extreme-when-bench",
                    help="HF dataset name (default) or local JSON path")
    ap.add_argument("--video-dir", required=True, help="dir containing {video_id}.mp4")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--num-frames", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.5-9b")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
