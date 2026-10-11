#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native HTTP render IDs with an independent Hugging Face tokenizer.

This process does not import the cache adapter. Qwen3 text-only chat semantics
are checked here; this is not a multimodal or generation-quality evaluator.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from verify_serving import cases, post, token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--cached", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    corpus = list(cases(args.model))
    tools_case = dict(corpus[-1][1], tool_choice="auto")
    corpus.extend(
        [
            ("tools-auto", tools_case),
            (
                "thinking-disabled",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": "你好 👋"}],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ),
            (
                "empty-user",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": ""}],
                },
            ),
        ]
    )
    result = {
        "scope": "qwen3_text_native_http_vs_independent_hf",
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
        "npu_inference": "not_evaluated",
        "generation_quality": "not_evaluated",
        "records": [],
    }
    try:
        for name, payload in corpus:
            # Match the default native render policy: keep tools for tool_choice=none.
            tools = payload.get("tools")
            expected = tokenizer.apply_chat_template(
                payload["messages"],
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
                **payload.get("chat_template_kwargs", {}),
            )
            if isinstance(expected, Mapping):
                expected = expected["input_ids"]
            if hasattr(expected, "tolist"):
                expected = expected.tolist()
            for mode, base in [("off", args.baseline), ("on", args.cached)]:
                observed = token_ids(
                    post(base, "/v1/chat/completions/render", payload).get("token_ids")
                )
                result["records"].append(
                    {
                        "case": name,
                        "mode": mode,
                        "input": payload,
                        "expected_ids": expected,
                        "observed_ids": observed,
                        "equal": expected == observed,
                    }
                )
                if expected != observed:
                    raise ValueError(f"full token IDs differ: {name}/{mode}")
        result["passed"] = True
    except Exception as error:
        result.update(passed=False, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
