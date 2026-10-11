#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch CPU rendering with the fixed Ascend segment-cache implementation.

The adapter attaches a cache to each renderer's tokenizer. It does not load
weights or initialize the Ascend platform; VLLM_PLUGINS must exclude Ascend.
The paired render launcher must preserve additional_config in VllmConfig.
"""

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from launch_backend import inspect_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-source", type=Path, required=True)
    parser.add_argument("--cache-stats-output", type=Path, required=True)
    args, command = parser.parse_known_args()
    if command[:2] != ["launch", "render"]:
        parser.error("only the CPU launch render command is supported")
    if "VLLM_PLUGINS" not in os.environ or "ascend" in os.environ["VLLM_PLUGINS"]:
        parser.error("set VLLM_PLUGINS='' to exclude the Ascend platform")
    if args.cache_stats_output.exists():
        parser.error("use a new stats path to preserve previous evidence")

    contract = json.loads(Path(__file__).with_name("source_contract.json").read_text())
    inspect_backend(
        {
            "package_files": {
                "vllm": {
                    **contract["package_files"]["vllm"],
                    "entrypoints/launchers/render/entry.py": contract[
                        "cpu_render_launcher_sha256"
                    ],
                }
            }
        }
    )
    expected = contract["package_files"]["vllm_ascend"][
        "patch/platform/patch_tokenizer_cache.py"
    ]
    if hashlib.sha256(args.cache_source.read_bytes()).hexdigest() != expected:
        raise ValueError("cache source does not match the fixed Ascend PR head")

    spec = importlib.util.spec_from_file_location(
        "cpu_segment_cache", args.cache_source
    )
    cache_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache_module)
    # CPU capacity is read by the factory below, without Ascend initialization.
    cache_module._configured_capacity_gb = lambda: 0

    from vllm.entrypoints.launchers.render import app_state

    original_factory = app_state.renderer_from_config
    caches = []

    def renderer_factory(config):
        renderer = original_factory(config)
        capacity = config.additional_config.get("tokenizer_cache_gb", 0)
        if type(capacity) is not int or capacity < 0:
            raise ValueError("tokenizer_cache_gb must be a non-negative integer")
        if capacity:
            cache = cache_module.IncrementalTokenizerCache(renderer.tokenizer, capacity)
            if not cache.enabled:
                raise ValueError(
                    "requested cache is disabled: " + str(cache.disabled_reason)
                )
            cache_module._install_cache(renderer.tokenizer, cache)
            caches.append(cache)
        return renderer

    def save_stats():
        args.cache_stats_output.write_text(
            json.dumps(
                {
                    "scope": "cpu_renderer_adapter",
                    "renderer_cache_instances": len(caches),
                    "cache_hits": sum(cache.stat().hits for cache in caches),
                    "cache_lookups": sum(cache.stat().total for cache in caches),
                    "npu_inference": "not_evaluated",
                },
                indent=2,
            )
            + "\n"
        )

    app_state.renderer_from_config = renderer_factory
    atexit.register(save_stats)
    sys.argv = [sys.argv[0]] + command
    from vllm.entrypoints.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
