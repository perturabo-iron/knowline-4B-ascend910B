#!/usr/bin/env python3
"""Clean functional & latency verification script for KnowLine-4B on Ascend 910B4."""
import argparse
import statistics
import sys
import time
import requests


MULTI_Q_PAYLOAD = {
    "model": "m",
    "state": "Customer: my order arrived broken, I want my money back.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Should the agent offer a refund?"},
        "tone": {
            "type": "choice",
            "instructions": "Customer tone?",
            "criteria": {"angry": "Angry", "neutral": "Neutral", "happy": "Happy"},
        },
    },
}

SINGLE_Q_PAYLOAD = {
    "model": "m",
    "state": "The invoice total is $120 and the customer paid $120 via credit card.",
    "questions": {
        "paid": {"type": "noul", "instructions": "Has the customer paid the invoice in full?"},
    },
}


def percentile(values, p):
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def main():
    ap = argparse.ArgumentParser(description="Verify KnowLine-4B service correctness and latency.")
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="Base URL of KnowLine front-end")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup requests before timing")
    ap.add_argument("--runs", type=int, default=20, help="Timed benchmark runs")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    sess = requests.Session()

    # 1. Health check
    r = sess.get(f"{base}/health", timeout=10)
    r.raise_for_status()
    print(f"[PASS] GET {base}/health -> {r.status_code}")

    # 2. Single-question sanity check
    r1 = sess.post(f"{base}/v1/systemone", json=SINGLE_Q_PAYLOAD, timeout=60)
    r1.raise_for_status()
    d1 = r1.json()
    p_paid = d1["answers"]["paid"]["noul"]
    assert 0.5 < p_paid <= 1.0, f"Unexpected single-question score: {d1}"
    assert d1["usage"]["output_tokens"] == 1
    print(f"[PASS] Single-question check: paid.noul={p_paid:.6f}, usage={d1['usage']}")

    # 3. Multi-question warmup + functional check
    d2 = None
    for _ in range(args.warmup):
        r2 = sess.post(f"{base}/v1/systemone", json=MULTI_Q_PAYLOAD, timeout=60)
        r2.raise_for_status()
        d2 = r2.json()
    assert d2 is not None
    refund_p = d2["answers"]["refund"]["noul"]
    tone_ans = d2["answers"]["tone"]
    prob_sum = sum(tone_ans["probabilities"].values())
    assert refund_p > 0.9, f"Unexpected refund score: {refund_p}"
    assert tone_ans["choice"] == "angry", f"Unexpected tone choice: {tone_ans}"
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1: {prob_sum}"
    print(
        f"[PASS] Multi-question check: refund.noul={refund_p:.6f}, "
        f"tone={tone_ans['choice']} (angry={tone_ans['probabilities']['angry']:.6f}), "
        f"usage={d2['usage']}"
    )

    # 4. Timed latency benchmark
    times_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        r = sess.post(f"{base}/v1/systemone", json=MULTI_Q_PAYLOAD, timeout=60)
        r.raise_for_status()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    med_ms = statistics.median(times_ms)
    min_ms = min(times_ms)
    p95_ms = percentile(times_ms, 95)
    max_ms = max(times_ms)
    print(
        f"[BENCH] Multi-question (n={args.runs}): "
        f"mean={mean_ms:.2f} ms | median={med_ms:.2f} ms | "
        f"min={min_ms:.2f} ms | p95={p95_ms:.2f} ms | max={max_ms:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
