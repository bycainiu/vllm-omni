# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Assemble a full MiniMax H3 model directory around a new FP8 transformer."""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
from pathlib import Path

from experiments.minimax_h3_offline_fp8.export import export_checkpoint


def build_candidate(source_model: Path, output_model: Path, *, num_threads: int | None = None) -> Path:
    source = source_model.resolve()
    output = output_model.absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing model directory: {output}")
    if not (source / "transformer").is_dir():
        raise FileNotFoundError(f"source model has no transformer directory: {source}")

    stage = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")
    stage.mkdir(parents=True)
    try:
        for entry in source.iterdir():
            if entry.name == "transformer":
                continue
            link_target = os.path.relpath(entry, stage)
            (stage / entry.name).symlink_to(link_target, target_is_directory=entry.is_dir())

        export_checkpoint(
            source / "transformer",
            stage / "transformer",
            num_threads=num_threads,
        )
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_model", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--num-threads", type=int, default=None)
    args = parser.parse_args()
    print(build_candidate(args.source_model, args.output_model, num_threads=args.num_threads))


if __name__ == "__main__":
    main()
