"""KnowLine /v1/systemone on vLLM 0.30.0: the model repo's own knowline_server.py, with a vLLM backend in place of SGLang.
Nothing in the model repo changes, and this file can sit anywhere. For deployment; our submitted runs use SGLang.

    vllm serve <model dir> --served-model-name m --enable-scale-out --max-logprobs 128 --port 9080 ...
    python knowline_vllm.py --model <model dir> --backend vllm --url http://127.0.0.1:9080 [knowline_server.py options]
    (serve_knowline_vllm.sh runs both, with the engine flags and front-end options of the model's serve_knowline.sh)

The front end (rendering, labels, read twice, softmax over the labels, answers, the HTTP server and all its options) is the
knowline_server.py in --server-dir (default: the --model dir), unchanged. Only the backend differs:
  - vLLM 0.30.0 returns chosen token ids' logprobs only through SamplingParams.logprob_token_ids, which no OpenAI route
    takes. The token-in/token-out route /inference/v1/generate (registered by `vllm serve --enable-scale-out`) takes raw
    SamplingParams, so the prompt is tokenized here with the model's tokenizer (no special tokens added: the chat
    template writes them) and sent as token ids.
  - Label logprobs: one greedy token with logprob_token_ids = the labels' ids, which reads log-softmax over the whole
    vocabulary before temperature (vLLM's default --logprobs-mode raw_logprobs). vLLM 0.30.0 takes at most 128 ids per
    request (MAX_LOGPROB_TOKEN_IDS), so a question with more than 127 options is read in several requests on the same
    prompt; full-vocabulary logprobs join exactly, and the repeated prompt is a prefix-cache hit.
  - Box questions (servers that have them): greedy decoding with the sampled tokens' logprobs, returned in the SGLang
    /generate shape that knowline_server.py reads.
  - Images: sent as content_parts (data: or http(s) URLs); vLLM expands the image placeholders of the token prompt.
Requirements: requests and transformers (vLLM's environment has both).
Licence of this file: MIT.
"""
import argparse
import base64
import math
import mimetypes
import os
import sys
import threading
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

CHUNK = 127  # label ids per request: MAX_LOGPROB_TOKEN_IDS (128) minus the repeated last id (see VLLM.score)
MIME = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}


def _finite(values):
    return [v if v is not None and math.isfinite(v) else -math.inf for v in values]


def _content_part(m):
    """knowline_server media item (a source, or (kind, source)) -> a vLLM content part with a data: or http(s) URL."""
    kind, src = ("image", m) if isinstance(m, str) else m
    if not src.startswith(("data:", "http://", "https://")):
        if os.path.isfile(src):  # SGLang also takes a local path
            mime = mimetypes.guess_type(src)[0] or MIME[kind]
            src = f"data:{mime};base64,{base64.b64encode(Path(src).read_bytes()).decode()}"
        else:  # bare base64, as SGLang takes it; vLLM decodes the image whatever the declared type
            src = f"data:{MIME[kind]};base64,{src}"
    return {"type": f"{kind}_url", "url": src}


def _token_id(entry):
    return int(entry["token"].removeprefix("token_id:"))


class VLLM:
    """vLLM 0.30.0 /inference/v1/generate with max_tokens=1 and logprob_token_ids: one prefill, the label logprobs of the
    next token. Same methods as knowline_server.SGLang."""

    def __init__(self, url, model, timeout=120):
        from transformers import AutoTokenizer
        self.url, self.timeout, self.http = url.rstrip("/"), timeout, requests.Session()
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.http.mount("http://", adapter)
        self.http.mount("https://", adapter)
        self.tok, self.lock = AutoTokenizer.from_pretrained(model), threading.Lock()
        # HOST-3 controls (env-tunable for A/B testing):
        # - KNOWLINE_WARM_MIN_TOKENS: skip warm(prefix) when prefix_tokens < threshold
        #   (Qwen3.5-4B default effective mamba_block_size is 640 tokens; shorter prefixes cache 0 blocks).
        # - KNOWLINE_BATCH_SCORE: submit multi-question scores as one atomic /inference/v1/generate_batch call.
        self.warm_min_tokens = int(os.getenv("KNOWLINE_WARM_MIN_TOKENS", "640"))
        self.batch_score_enabled = os.getenv("KNOWLINE_BATCH_SCORE", "1") != "0"
        self._batch_endpoint_available = self.batch_score_enabled

    def _post_ids(self, ids, content_parts, sampling_params):
        body = {"token_ids": ids, "sampling_params": sampling_params}
        if content_parts:
            body["content_parts"] = content_parts
        r = self.http.post(f"{self.url}/inference/v1/generate", json=body, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        return out["choices"][0], (out.get("usage") or {}).get("prompt_tokens", len(ids))

    def _post(self, text, media, sampling_params):
        with self.lock:  # one tokenizer shared by the scoring threads
            ids = self.tok.encode(text, add_special_tokens=False)
        content_parts = [_content_part(m) for m in media] if media else None
        return self._post_ids(ids, content_parts, sampling_params)

    def warm(self, prefix, media):
        with self.lock:
            ids = self.tok.encode(prefix, add_special_tokens=False)
        if not media and self.warm_min_tokens > 0 and len(ids) < self.warm_min_tokens:
            # HOST-3: prefix shorter than one Mamba cache block caches 0 blocks; skip wasted step.
            return
        content_parts = [_content_part(m) for m in media] if media else None
        self._post_ids(ids, content_parts, {"max_tokens": 1, "temperature": 0.0})

    def score(self, text, media, ids):
        got, n = {}, 0
        with self.lock:
            tok_ids = self.tok.encode(text, add_special_tokens=False)
        content_parts = [_content_part(m) for m in media] if media else None
        for k in range(0, len(ids), CHUNK):
            part = ids[k:k + CHUNK]
            # vLLM lists the sampled token first and keeps `logprobs` entries; with the last id repeated (one entry
            # once deduplicated) every label still fits when the sampled token is not a label.
            choice, n = self._post_ids(tok_ids, content_parts, {"max_tokens": 1, "temperature": 0.0,
                                                                 "logprobs": len(part) + 1,
                                                                 "logprob_token_ids": part + part[-1:]})
            got.update((_token_id(t), t["logprob"]) for t in choice["logprobs"]["content"][0]["top_logprobs"])
        return _finite([got.get(i) for i in ids]), n

    def score_batch(self, items_prompts_and_ids, media):
        """HOST-3: batch-tokenize all questions in one call and submit atomically to /inference/v1/generate_batch."""
        texts = [p for p, _ in items_prompts_and_ids]
        with self.lock:
            all_tok_ids = self.tok(texts, add_special_tokens=False)["input_ids"]
        content_parts = [_content_part(m) for m in media] if media else None
        bodies = []
        mapping = []  # (item_idx, label_chunk)
        for item_idx, ((_, label_ids), tok_ids) in enumerate(zip(items_prompts_and_ids, all_tok_ids)):
            for k in range(0, len(label_ids), CHUNK):
                part = label_ids[k:k + CHUNK]
                body = {
                    "token_ids": tok_ids,
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "logprobs": len(part) + 1,
                        "logprob_token_ids": part + part[-1:],
                    },
                }
                if content_parts:
                    body["content_parts"] = content_parts
                bodies.append(body)
                mapping.append((item_idx, len(tok_ids)))

        if self._batch_endpoint_available and len(bodies) > 1:
            r = self.http.post(f"{self.url}/inference/v1/generate_batch", json=bodies, timeout=self.timeout)
            if r.status_code == 404:
                self._batch_endpoint_available = False
            else:
                r.raise_for_status()
                outs = r.json()["results"]
                gots = [{} for _ in items_prompts_and_ids]
                ns = [0] * len(items_prompts_and_ids)
                for (item_idx, tok_len), out in zip(mapping, outs):
                    choice = out["choices"][0]
                    ns[item_idx] = (out.get("usage") or {}).get("prompt_tokens", tok_len)
                    gots[item_idx].update(
                        (_token_id(t), t["logprob"]) for t in choice["logprobs"]["content"][0]["top_logprobs"]
                    )
                return [
                    (_finite([gots[i].get(lid) for lid in label_ids]), ns[i])
                    for i, (_, label_ids) in enumerate(items_prompts_and_ids)
                ]

        # Fallback: concurrent pre-tokenized requests over keep-alive pool
        from concurrent.futures import ThreadPoolExecutor

        def _send_one(body_and_len):
            body, tok_len = body_and_len
            resp = self.http.post(f"{self.url}/inference/v1/generate", json=body, timeout=self.timeout)
            resp.raise_for_status()
            out = resp.json()
            return out["choices"][0], (out.get("usage") or {}).get("prompt_tokens", tok_len)

        with ThreadPoolExecutor(max_workers=min(16, len(bodies))) as ex:
            outs = list(ex.map(_send_one, [(b, tl) for b, (_, tl) in zip(bodies, mapping)]))
        gots = [{} for _ in items_prompts_and_ids]
        ns = [0] * len(items_prompts_and_ids)
        for (item_idx, _), (choice, n) in zip(mapping, outs):
            ns[item_idx] = n
            gots[item_idx].update(
                (_token_id(t), t["logprob"]) for t in choice["logprobs"]["content"][0]["top_logprobs"]
            )
        return [
            (_finite([gots[i].get(lid) for lid in label_ids]), ns[i])
            for i, (_, label_ids) in enumerate(items_prompts_and_ids)
        ]

    def generate(self, text, media, max_new_tokens=32):
        """Box: greedy decoding, the logprob of every output token (logprobs 0 = the sampled token's own), as SGLang's
        /generate response with return_text_in_logprobs."""
        choice, n = self._post(text, media, {"max_tokens": max_new_tokens, "temperature": 0.0, "logprobs": 0})
        content = choice["logprobs"]["content"]
        ids = choice.get("token_ids") or [_token_id(t) for t in content]
        with self.lock:
            text_out = self.tok.decode(ids, skip_special_tokens=True)
            pieces = [self.tok.decode([i]) for i in ids]
        return {"text": text_out,
                "meta_info": {"finish_reason": {"type": choice.get("finish_reason")}, "prompt_tokens": n,
                              "output_token_logprobs": [[t["logprob"], i, p] for t, i, p in zip(content, ids, pieces)]}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    p.add_argument("--server-dir", help="directory with knowline_server.py and knowline_layout.py (default: --model)")
    p.add_argument("--model", required=True, help="model dir (tokenizer + chat template; the weights are vLLM's)")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help="hf: knowline_server.py's own transformers backend")
    p.add_argument("--url", default="http://127.0.0.1:9080", help="vLLM server (vllm serve ... --enable-scale-out)")
    p.add_argument("--backend-timeout", type=float, default=120, help="seconds per vLLM request")
    a, rest = p.parse_known_args()
    server_dir = Path(a.server_dir or a.model)
    if not (server_dir / "knowline_server.py").is_file():
        p.error(f"no knowline_server.py in {server_dir}; pass --server-dir")
    sys.path.insert(0, str(server_dir.resolve()))
    import knowline_server as ks
    ks.SGLang = lambda url, **kw: VLLM(url, a.model, a.backend_timeout)  # main() builds SGLang(a.url) for --backend sglang
    # HOST-3: route multi-question KnowLine.run() through VLLM.score_batch() without modifying knowline_server.py.
    _orig_knowline_run = ks.KnowLine.run

    def _batched_knowline_run(self, state, questions):
        if not (getattr(self.backend, "batch_score_enabled", False) and hasattr(self.backend, "score_batch") and len(questions) > 1):
            return _orig_knowline_run(self, state, questions)
        if not 1 <= len(questions) <= ks.MAX_QUESTIONS:
            raise ValueError(f"1..{ks.MAX_QUESTIONS} questions, got {len(questions)}")
        try:
            if self.layout is None:
                prefix, prompts, media = ks.render(self.processor, state, questions, self.labels)
            else:
                from knowline_layout import render_request
                r = render_request({"state": state, "questions": questions}, self.processor, self.layout, labels=self.labels)
                prefix, media = r["prefix"], r["media"]
                prompts = {qid: r["prompts"][qid][0] for qid in questions}
        except ValueError as exc:
            if "criteria" in str(exc) or "options" in str(exc):
                raise ValueError(f"{exc} (too many options per choice for this label set)") from exc
            raise
        items = list(prompts.items())
        self.backend.warm(prefix, media)
        results = self.backend.score_batch(
            [(it[1][0], self.ids[:len(it[1][1])]) for it in items],
            media,
        )
        answers = {
            qid: ks.answer(questions[qid], keys, ks.softmax(row, self.T_by_type.get(questions[qid].get("type"), self.T)))
            for (qid, (_, keys)), (row, _) in zip(items, results)
        }
        return answers, {"input_tokens": sum(n for _, n in results), "output_tokens": len(items)}

    ks.KnowLine.run = _batched_knowline_run
    sys.argv = [ks.__file__, *rest, "--model", a.model, "--url", a.url, "--backend", "sglang" if a.backend == "vllm" else "hf"]
    if a.backend == "vllm":
        print(f"knowline_vllm: {ks.__file__} with the vLLM backend at {a.url} (the line below says sglang)", flush=True)
    ks.main()


if __name__ == "__main__":
    main()
