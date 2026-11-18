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

"""Checkpoint-owned ACT computation for full numerical pipeline export."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.envs.utils import prepare_canonical_observation
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)

DEFAULT_ACT_CHECKPOINT = "aadarshram/act_pusht"
DEFAULT_ACT_REVISION = "864fd80821cccec9c6783e036684c46be52eda2a"


def _source_feature_name(name: str, preprocessor: PolicyProcessorPipeline) -> str:
    """Follow explicit renames backwards from a model feature to its raw input."""
    for step in reversed(preprocessor.steps):
        if not isinstance(step, RenameObservationsProcessorStep):
            continue
        sources = [source for source, target in step.rename_map.items() if target == name]
        if len(sources) > 1:
            raise ValueError(f"Ambiguous input semantics for {name!r}: rename sources {sources}")
        if sources:
            name = sources[0]
        elif name in step.rename_map:
            raise ValueError(f"Cannot resolve raw input {name!r}: the preprocessor renames it away")
    return name


class ACTPipeline:
    """Run saved ACT processors around chunk prediction, without action scheduling.

    Observations are canonical LeRobot keys with batched HWC uint8 images and
    batched state vectors. Only the standard, stateless processor steps are
    accepted: chunk postprocessing must equal postprocessing each action.
    """

    def __init__(
        self,
        policy: ACTPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        fps: float,
        provenance: dict[str, Any],
    ) -> None:
        """Validate the stateless ACT path and describe its deployment boundary."""
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("The control rate must be finite and positive")
        config = policy.config
        if not isinstance(config, ACTConfig):
            raise ValueError("LEAPP ACT export requires an ACT configuration")
        if config.temporal_ensemble_coeff is not None or config.n_obs_steps != 1:
            raise ValueError(
                "LEAPP ACT export requires one observation step and temporal ensembling disabled"
            )
        if config.use_amp or config.dtype not in (None, torch.float32):
            raise ValueError("LEAPP ACT export currently supports float32 inference without AMP")
        allowed_pre = {
            RenameObservationsProcessorStep,
            AddBatchDimensionProcessorStep,
            DeviceProcessorStep,
            NormalizerProcessorStep,
            IdentityProcessorStep,
        }
        allowed_post = {UnnormalizerProcessorStep, DeviceProcessorStep, IdentityProcessorStep}
        for direction, processor, allowed in (
            ("pre", preprocessor, allowed_pre),
            ("post", postprocessor, allowed_post),
        ):
            if processor.before_step_hooks or processor.after_step_hooks:
                raise ValueError(f"LEAPP ACT export does not support {direction}processor hooks")
            for step in processor.steps:
                if type(step) not in allowed:
                    raise ValueError(
                        f"Unsupported ACT {direction}processor {type(step).__name__}; "
                        "use export_pipeline with an explicitly qualified numerical callable"
                    )
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.provenance = provenance
        self._leapp_deployment = {
            "action_output": "action_chunk",
            "n_action_steps": config.n_action_steps,
            "control_fps": fps,
            "provenance": provenance,
        }
        self._leapp_semantics: dict[str, dict[str, dict[str, Any]]] = {
            "inputs": {
                _source_feature_name(name, preprocessor): {"kind": "image"} for name in config.image_features
            },
            "outputs": {"action_chunk": {"axes": [None, {"kind": "time", "period_ms": 1000.0 / fps}, None]}},
        }

    @torch.no_grad()
    def __call__(self, observations: dict[str, np.ndarray | Tensor]) -> dict[str, Tensor]:
        """Convert raw observations and return a fully postprocessed action chunk."""
        # Keep the established image/layout conversion in the captured computation.
        arrays = {
            name: value.detach().cpu().numpy() if isinstance(value, Tensor) else value
            for name, value in observations.items()
        }
        batch = self.preprocessor(prepare_canonical_observation(arrays))
        actions = self.policy.predict_action_chunk(batch)
        return {"action_chunk": self.postprocessor(actions)}


def load_act_pipeline(
    checkpoint: str | Path,
    revision: str | None = None,
    *,
    fps: float = 10.0,
) -> ACTPipeline:
    """Strictly load ACT weights and saved processors on CPU for export.

    Mutable Hub refs are resolved once so configuration, weights, and statistics
    all come from the same immutable revision. Local paths are used unchanged.
    """
    source = str(checkpoint)
    resolved_revision = revision
    if not Path(source).is_dir() and not (revision and re.fullmatch(r"[0-9a-f]{40}", revision)):
        resolved_revision = HfApi().model_info(source, revision=revision).sha
    config = PreTrainedConfig.from_pretrained(
        source,
        revision=resolved_revision,
        cli_overrides=["--device=cpu", "--use_amp=false"],
    )
    if not isinstance(config, ACTConfig):
        raise ValueError(f"LEAPP checkpoint export currently supports ACT, got {config.type!r}")
    if config.temporal_ensemble_coeff is not None or config.n_obs_steps != 1:
        raise ValueError("LEAPP ACT export requires one observation step and temporal ensembling disabled")
    # The checkpoint supplies every backbone parameter; avoid an unused ImageNet download.
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(source, config=config, revision=resolved_revision, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=source,
        pretrained_revision=resolved_revision,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return ACTPipeline(
        policy,
        preprocessor,
        postprocessor,
        fps=fps,
        provenance={"checkpoint": source, "checkpoint_revision": resolved_revision},
    )
