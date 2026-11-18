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

"""Exercise queue scheduling and input ownership using a real exported ONNX pipeline."""

import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("leapp")

from lerobot.export.exporter import export_pipeline
from lerobot.export.runtime import LeappRunner


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    def pipeline(observation):
        state = observation["observation.state"]
        return {"action_chunk": torch.stack([state + i for i in (1, 2, 3)], dim=1)}

    pipeline._leapp_deployment = {"action_output": "action_chunk", "n_action_steps": 2, "control_fps": 10.0}
    samples = [{"observation.state": torch.full((1, 2), float(i))} for i in range(5)]
    return export_pipeline(pipeline, samples, tmp_path_factory.mktemp("runtime_export"))


def observation(value=0.0):
    return {"observation.state": np.full((1, 2), value, dtype=np.float32)}


def test_queue_refills_at_n_action_steps_and_reset_discards_pending_actions(bundle):
    runner = LeappRunner(bundle)
    for cycle in range(3):
        value = float(cycle * 10)
        torch.testing.assert_close(runner.select_action(observation(value)), torch.full((1, 2), value + 1))
        torch.testing.assert_close(
            runner.select_action(observation(value + 100)), torch.full((1, 2), value + 2)
        )
    runner.select_action(observation(50.0))
    runner.reset()
    torch.testing.assert_close(runner.select_action(observation(80.0)), torch.full((1, 2), 81.0))


@pytest.mark.parametrize("as_numpy", [False, True])
def test_runtime_owns_inputs_and_returned_outputs(bundle, as_numpy):
    runner = LeappRunner(bundle)
    source = torch.full((1, 2), 7.0)
    value = source.numpy() if as_numpy else source
    chunk = runner.predict_action_chunk({"observation.state": value})
    action = runner.select_action({"observation.state": value})
    expected_chunk, expected_action = chunk.clone(), action.clone()
    runner.predict_action_chunk(observation(-20.0))
    runner.reset()
    torch.testing.assert_close(source, torch.full_like(source, 7.0))
    torch.testing.assert_close(chunk, expected_chunk)
    torch.testing.assert_close(action, expected_action)


@pytest.mark.parametrize(
    "invalid, message",
    [
        ({}, "Observation keys"),
        ({"observation.state": np.zeros((2,), dtype=np.float32)}, "shape="),
        ({"observation.state": np.zeros((1, 2), dtype=np.float64)}, "dtype="),
        ({"observation.state": [[0.0, 0.0]]}, "NumPy array or Torch tensor"),
        ({"observation.state": np.full((1, 2), np.nan, dtype=np.float32)}, "non-finite"),
    ],
)
def test_invalid_observations_rejected_even_when_action_is_queued(bundle, invalid, message):
    runner = LeappRunner(bundle)
    runner.select_action(observation())
    with pytest.raises((ValueError, TypeError), match=message):
        runner.select_action(invalid)
    torch.testing.assert_close(runner.select_action(observation()), torch.full((1, 2), 2.0))


@pytest.mark.parametrize("mutation", ["missing_schedule", "too_many_steps", "feedback", "aliases"])
def test_unsupported_deployment_contracts_fail_clearly(bundle, mutation):
    description = yaml.safe_load(bundle.read_text())
    config = description["pipeline"]["configs"]["lerobot"]
    if mutation == "missing_schedule":
        del config["n_action_steps"]
    elif mutation == "too_many_steps":
        config["n_action_steps"] = 4
    elif mutation == "feedback":
        description["pipeline"]["feedback_flow"] = {"policy/action_chunk": ["policy/observation_state"]}
    else:
        config["input_aliases"] = {"observation.state": "policy/missing"}
    modified = bundle.with_name(f"{mutation}.yaml")
    modified.write_text(yaml.safe_dump(description))
    with pytest.raises(ValueError, match="n_action_steps|feedback|input_aliases"):
        runner = LeappRunner(modified)
        runner.select_action(observation())


def test_relocated_bundle_runs_without_policy_or_processor_imports(bundle, tmp_path):
    destination = tmp_path / "relocated"
    shutil.copytree(bundle.parent, destination)
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class BlockPolicySources(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(('lerobot.policies', 'lerobot.processor')):
                    raise ImportError('policy source is unavailable')
        sys.meta_path.insert(0, BlockPolicySources())
        import torch
        from lerobot.export.runtime import LeappRunner
        from lerobot.scripts.lerobot_eval_leapp import LeappEvalConfig
        runner = LeappRunner(sys.argv[1])
        action = runner.select_action({'observation.state': torch.full((1, 2), 4.0)})
        torch.testing.assert_close(action, torch.full((1, 2), 5.0))
        """
    )
    subprocess.run([sys.executable, "-c", script, str(destination / bundle.name)], check=True, timeout=60)
