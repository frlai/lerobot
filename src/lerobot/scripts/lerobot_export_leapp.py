#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a saved ACT policy, including its processors, using LEAPP."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from lerobot.configs import parser
from lerobot.envs import EnvConfig, PushtEnvConfig, make_task_env
from lerobot.export.act import DEFAULT_ACT_CHECKPOINT, DEFAULT_ACT_REVISION, load_act_pipeline
from lerobot.export.exporter import export_pipeline
from lerobot.utils.utils import init_logging


@dataclass
class LeappExportConfig:
    checkpoint_path: str = DEFAULT_ACT_CHECKPOINT
    revision: str | None = None
    output_dir: Path = Path("outputs/export/act_pusht")
    env: EnvConfig = field(default_factory=PushtEnvConfig)
    backend: str = "onnx"

    def __post_init__(self) -> None:
        """Pin the example checkpoint without applying its revision to other models."""
        if self.revision is None and self.checkpoint_path == DEFAULT_ACT_CHECKPOINT:
            self.revision = DEFAULT_ACT_REVISION


@parser.wrap()
def export_main(cfg: LeappExportConfig) -> Path:
    """Capture five seeded real observations and export their complete numerical path."""
    if not isinstance(cfg.env, PushtEnvConfig):
        raise ValueError(
            "The checkpoint export CLI currently qualifies PushT; use export_pipeline for other inputs"
        )
    pipeline = load_act_pipeline(cfg.checkpoint_path, cfg.revision, fps=cfg.env.fps)
    env = make_task_env(cfg.env, n_envs=1)
    try:
        samples = [env.reset(seed=seed)[0] for seed in range(5)]
        descriptor = env.descriptor
        for feature in descriptor.features:
            if feature.names:
                pipeline._leapp_semantics["inputs"].setdefault(feature.name, {})["axes"] = [
                    None,
                    {"kind": "element", "names": list(feature.names)},
                ]
        names = descriptor.action_feature.names
        if names:
            pipeline._leapp_semantics["outputs"]["action_chunk"]["axes"][-1] = {
                "kind": "element",
                "names": list(names),
            }
        pipeline.provenance["environment_semantics"] = descriptor.semantics
        return export_pipeline(pipeline, samples, cfg.output_dir, backend=cfg.backend)
    finally:
        env.close()


def main() -> None:
    init_logging()
    # parser.wrap supplies the configuration when the entry point has no arguments.
    cast(Callable[[], Path], export_main)()


if __name__ == "__main__":
    main()
