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

import shutil

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("leapp")

from leapp import InferenceManager

from lerobot.export.annotations import export_semantics
from lerobot.export.exporter import export_pipeline


@export_semantics(inputs={"observation.image": {"kind": "image"}})
def numerical_pipeline(observation):
    image = torch.from_numpy(observation["observation.image"]).permute(0, 3, 1, 2).float() / 255
    normalized = (image - 0.5) / 0.2
    state = torch.from_numpy(observation["observation.state"])
    prediction = normalized.mean(dim=(1, 2, 3)).unsqueeze(-1) + state * 2
    return {"action_chunk": (prediction * 3 + 4).unsqueeze(1).repeat(1, 3, 1)}


def sample(seed):
    rng = np.random.default_rng(seed)
    return {
        "observation.image": rng.integers(0, 256, (1, 8, 8, 3), dtype=np.uint8),
        "observation.state": rng.normal(size=(1, 2)).astype(np.float32),
    }


def test_numerical_pipeline_relocated_fresh_inputs(tmp_path):
    """Exercise raw NumPy capture, one-model composition, aliases, strict export and relocated replay."""
    original = sample(0)
    saved = {key: value.copy() for key, value in original.items()}
    reordered = dict(reversed(list(sample(1).items())))
    bundle = export_pipeline(numerical_pipeline, [original, reordered, sample(2)], tmp_path)
    for key in saved:
        np.testing.assert_array_equal(original[key], saved[key])
    description = yaml.safe_load(bundle.read_text())
    assert len(description["models"]) == 1
    assert description["models"]["policy"]["inputs"][0]["kind"] == "image"
    config = description["pipeline"]["configs"]["lerobot"]
    assert config["input_aliases"]["observation.image"] == "policy/observation_image"
    copied = tmp_path / "relocated"
    shutil.copytree(bundle.parent, copied)
    manager = InferenceManager(str(copied / bundle.name))
    for seed in range(10, 30):
        observation = sample(seed)
        expected = numerical_pipeline(observation)["action_chunk"]
        actual = manager.run_policy(
            {port: torch.from_numpy(observation[key].copy()) for key, port in config["input_aliases"].items()}
        )["policy/action_chunk"]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    with pytest.raises(FileExistsError):
        export_pipeline(numerical_pipeline, [sample(0)], tmp_path)


def test_capture_failure_restores_patches_and_removes_incomplete_bundle(tmp_path):
    from_numpy = torch.from_numpy

    def fails(observation):
        torch.from_numpy(observation["observation.image"])
        raise ValueError("intentional capture failure")

    with pytest.raises(ValueError, match="intentional capture failure"):
        export_pipeline(fails, [sample(0)], tmp_path)
    assert torch.from_numpy is from_numpy
    assert not (tmp_path / "policy").exists()
    # A subsequent real capture in the same process must still work.
    assert export_pipeline(numerical_pipeline, [sample(1), sample(2)], tmp_path).is_file()


@pytest.mark.parametrize("bad_samples", [[], [{"x": "text"}], [{"x": torch.ones(2)}, {"x": torch.ones(3)}]])
def test_invalid_capture_contract_before_session(tmp_path, bad_samples):
    with pytest.raises((ValueError, TypeError)):
        export_pipeline(lambda x: x, bad_samples, tmp_path)
    assert not (tmp_path / "policy").exists()
