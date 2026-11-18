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

"""ACT checkpoint, saved statistics, and chunk/postprocessor compatibility."""

import shutil
from copy import deepcopy

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.utils import prepare_canonical_observation
from lerobot.export.act import ACTPipeline, load_act_pipeline
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import IdentityProcessorStep, RenameObservationsProcessorStep
from lerobot.scripts.lerobot_export_leapp import LeappExportConfig


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(41)
    config = ACTConfig(
        device="cpu",
        input_features={
            "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        pretrained_backbone_weights=None,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        chunk_size=3,
        n_action_steps=2,
    )
    policy = ACTPolicy(config).eval()
    stats = {
        "observation.image": {"mean": torch.tensor([0.4, 0.5, 0.6]), "std": torch.tensor([0.2, 0.3, 0.4])},
        "observation.state": {"mean": torch.tensor([1.0, -2.0]), "std": torch.tensor([2.0, 3.0])},
        "action": {"mean": torch.tensor([5.0, -6.0]), "std": torch.tensor([7.0, 8.0])},
    }
    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)
    path = tmp_path_factory.mktemp("act-checkpoint")
    config._save_pretrained(path)
    save_file(policy.state_dict(), path / "model.safetensors")
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path, policy, preprocessor, postprocessor


def observation():
    return {
        "observation.image": np.arange(32 * 32 * 3, dtype=np.uint8).reshape(1, 32, 32, 3),
        "observation.state": np.array([[3.0, 7.0]], dtype=np.float32),
    }


def test_checkpoint_processors_and_chunk_equivalence(checkpoint):
    """Round-trip saved weights/stats and compare chunk processing with native step processing."""
    path, reference, preprocessor, postprocessor = checkpoint
    pipeline = load_act_pipeline(path, fps=10)
    raw = observation()
    raw_copy = {name: value.copy() for name, value in raw.items()}
    with torch.no_grad():
        normalized = reference.predict_action_chunk(preprocessor(prepare_canonical_observation(raw)))
        expected = torch.stack([postprocessor(normalized[:, i]) for i in range(3)], dim=1)
        actual = pipeline(raw)["action_chunk"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert pipeline.policy.config.device == "cpu"
    assert pipeline._leapp_deployment["n_action_steps"] == 2
    assert pipeline._leapp_semantics["outputs"]["action_chunk"]["axes"][1]["period_ms"] == 100
    for name, value in raw.items():
        np.testing.assert_array_equal(value, raw_copy[name])
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(pipeline.policy.state_dict()[name], value, rtol=0, atol=0)


def test_strict_checkpoint_loading(checkpoint, tmp_path):
    path = tmp_path / "incomplete"
    shutil.copytree(checkpoint[0], path)
    state = load_file(path / "model.safetensors")
    del state["model.action_head.bias"]
    save_file(state, path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_act_pipeline(path)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_control_rate(checkpoint, fps):
    _, policy, preprocessor, postprocessor = checkpoint
    with pytest.raises(ValueError, match="control rate"):
        ACTPipeline(policy, preprocessor, postprocessor, fps=fps, provenance={})


def test_unqualified_postprocessor_rejected(checkpoint):
    class StatefulStep(IdentityProcessorStep):
        pass

    _, policy, preprocessor, _ = checkpoint
    _, postprocessor = make_act_pre_post_processors(policy.config)
    postprocessor.steps = [StatefulStep()]
    with pytest.raises(ValueError, match="Unsupported ACT postprocessor StatefulStep"):
        ACTPipeline(policy, preprocessor, postprocessor, fps=10, provenance={})


def test_hooks_and_temporal_ensemble_rejected(checkpoint):
    _, policy, _, postprocessor = checkpoint
    preprocessor, _ = make_act_pre_post_processors(policy.config)
    preprocessor.register_before_step_hook(lambda *_: None)
    with pytest.raises(ValueError, match="hooks"):
        ACTPipeline(policy, preprocessor, postprocessor, fps=10, provenance={})
    preprocessor.before_step_hooks.clear()
    policy.config.temporal_ensemble_coeff = 0.01
    try:
        with pytest.raises(ValueError, match="temporal ensembling"):
            ACTPipeline(policy, preprocessor, postprocessor, fps=10, provenance={})
    finally:
        policy.config.temporal_ensemble_coeff = None


def test_example_revision_does_not_leak_to_custom_checkpoint():
    assert LeappExportConfig().revision == "864fd80821cccec9c6783e036684c46be52eda2a"
    assert LeappExportConfig(checkpoint_path="another/model").revision is None
    assert LeappExportConfig(revision="main").revision == "main"


@pytest.fixture(scope="module")
def renamed_checkpoint(checkpoint, tmp_path_factory):
    path = tmp_path_factory.mktemp("act-renamed")
    shutil.copytree(checkpoint[0], path, dirs_exist_ok=True)
    preprocessor = deepcopy(checkpoint[2])
    preprocessor.steps[0].rename_map = {"observation.images.middle": "observation.image"}
    preprocessor.steps.insert(
        0,
        RenameObservationsProcessorStep(rename_map={"observation.images.raw": "observation.images.middle"}),
    )
    preprocessor.save_pretrained(path)
    return path


def renamed_observation():
    raw = observation()
    raw["observation.images.raw"] = raw.pop("observation.image")
    return raw


def test_saved_rename_chain_keeps_semantics_at_raw_boundary(checkpoint, renamed_checkpoint):
    pipeline = load_act_pipeline(renamed_checkpoint)
    reference = load_act_pipeline(checkpoint[0])
    torch.testing.assert_close(
        pipeline(renamed_observation())["action_chunk"],
        reference(observation())["action_chunk"],
        rtol=0,
        atol=0,
    )
    assert pipeline._leapp_semantics["inputs"] == {"observation.images.raw": {"kind": "image"}}
    both = renamed_observation()
    both["observation.image"] = both["observation.images.raw"].copy()
    with pytest.raises(ValueError, match="Both .*after rename_map"):
        pipeline(both)


def test_saved_rename_chain_exports_and_replays(renamed_checkpoint, tmp_path):
    """Exercise raw-key semantics through actual capture and a relocated-ready runtime contract."""
    pytest.importorskip("leapp")
    # Only this integration test needs the optional exporter/runtime dependencies.
    from lerobot.export.exporter import export_pipeline
    from lerobot.export.runtime import LeappRunner

    pipeline = load_act_pipeline(renamed_checkpoint)
    first, second = renamed_observation(), renamed_observation()
    second["observation.state"] += 0.5
    bundle = export_pipeline(pipeline, [first, second], tmp_path)
    runner = LeappRunner(bundle)
    fresh = renamed_observation()
    fresh["observation.images.raw"] = np.flip(fresh["observation.images.raw"], axis=2).copy()
    torch.testing.assert_close(
        runner.predict_action_chunk(fresh), pipeline(fresh)["action_chunk"], rtol=1e-4, atol=1e-4
    )
    assert "observation.images.raw" in runner.input_aliases
    assert "observation.image" not in runner.input_aliases


def test_ambiguous_image_rename_rejected(checkpoint):
    _, policy, preprocessor, postprocessor = checkpoint
    preprocessor = deepcopy(preprocessor)
    preprocessor.steps[0].rename_map = {
        "observation.images.left": "observation.image",
        "observation.images.right": "observation.image",
    }
    with pytest.raises(ValueError, match="Ambiguous input semantics"):
        ACTPipeline(policy, preprocessor, postprocessor, fps=10, provenance={})
