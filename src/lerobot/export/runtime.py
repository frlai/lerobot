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

"""Run exported numerical pipelines without their policy or processor classes."""

import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from lerobot.utils.import_utils import _leapp_available, require_package

if TYPE_CHECKING or _leapp_available:
    from leapp import InferenceManager


class LeappRunner:
    """Load a LEAPP bundle and schedule its fully processed action chunks.

    Numerical preprocessing and postprocessing execute in the exported model.
    This class only checks inputs, owns runtime buffers, and schedules actions.

    Args:
        bundle_path: Path to the exported LEAPP YAML description.
    """

    def __init__(self, bundle_path: str | Path):
        """Load the exported models and validate the deployment contract."""
        require_package("leapp", extra="leapp")
        self.bundle_path = Path(bundle_path)
        description = yaml.safe_load(self.bundle_path.read_text())
        if description["pipeline"].get("feedback_flow"):
            raise ValueError("LeappRunner does not support pipelines with feedback state.")
        self.manager = InferenceManager(str(self.bundle_path))
        self.config: dict[str, Any] = self.manager.pipeline.get("configs", {}).get("lerobot", {})
        if self.config.get("schema_version") != 1:
            raise ValueError("The bundle requires LeRobot deployment schema_version=1.")
        self.input_aliases = self.config.get("input_aliases", {})
        self.output_aliases = self.config.get("output_aliases", {})
        for direction, aliases, ports in (
            ("input", self.input_aliases, self.manager.inputs),
            ("output", self.output_aliases, self.manager.outputs),
        ):
            if (
                not isinstance(aliases, dict)
                or any(
                    not isinstance(key, str) or not isinstance(value, str) for key, value in aliases.items()
                )
                or len(set(aliases.values())) != len(aliases)
                or set(aliases.values()) != set(ports)
            ):
                raise ValueError(
                    f"Bundle {direction}_aliases must name each external {direction} exactly once."
                )
        self._input_specs = {
            name: self._port_description(port, "inputs") for name, port in self.input_aliases.items()
        }
        self._actions: deque[torch.Tensor] = deque()

    def _port_description(self, port: str, direction: str) -> dict[str, Any]:
        node_name, port_name = port.split("/")
        return next(
            description
            for description in self.manager.models[node_name][direction]
            if description["name"] == port_name
        )

    def _prepare_inputs(
        self, observation: Mapping[str, np.ndarray | torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if set(observation) != set(self.input_aliases):
            raise ValueError(
                f"Observation keys must be {sorted(self.input_aliases)}, got {sorted(observation)}."
            )
        inputs = {}
        for name, spec in self._input_specs.items():
            value = observation[name]
            shape = tuple(json.loads(spec["shape"]) if isinstance(spec["shape"], str) else spec["shape"])
            if not isinstance(value, (np.ndarray, torch.Tensor)):
                raise TypeError(f"Observation {name!r} must be a NumPy array or Torch tensor.")
            dtype = str(value.dtype).removeprefix("torch.")
            if tuple(value.shape) != shape or dtype != spec["dtype"]:
                raise ValueError(
                    f"Observation {name!r} requires shape={shape}, dtype={spec['dtype']}; "
                    f"got shape={tuple(value.shape)}, dtype={dtype}."
                )
            # InferenceManager.reset() zeroes retained inputs. Never retain caller-owned storage.
            tensor = (
                torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value.detach().clone()
            )
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Observation {name!r} contains non-finite values.")
            inputs[self.input_aliases[name]] = tensor
        return inputs

    def _action_contract(self) -> tuple[str, int, tuple[int, ...]]:
        output = self.config.get("action_output")
        count = self.config.get("n_action_steps")
        if output not in self.output_aliases or type(count) is not int or count <= 0:
            raise ValueError(
                "Action scheduling requires action_output and a positive n_action_steps in the bundle."
            )
        port = self.output_aliases[output]
        spec = self._port_description(port, "outputs")
        shape = tuple(json.loads(spec["shape"]) if isinstance(spec["shape"], str) else spec["shape"])
        if len(shape) != 3 or any(type(dim) is not int or dim <= 0 for dim in shape) or count > shape[1]:
            raise ValueError(
                "Action output must have fixed [batch, chunk, action] shape with n_action_steps <= chunk."
            )
        return port, count, shape

    def predict_action_chunk(self, observation: Mapping[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        """Return an owned, fully processed ``[batch, chunk, action]`` tensor.

        Args:
            observation: Raw observations matching the exported keys, shapes, and dtypes.

        Returns:
            An independent tensor that remains valid after subsequent inference or reset.
        """
        port, _, shape = self._action_contract()
        outputs = self.manager.run_policy(self._prepare_inputs(observation))
        chunk = outputs[port]
        if tuple(chunk.shape) != shape or not torch.isfinite(chunk).all():
            raise ValueError("Exported action output has an invalid shape or non-finite values.")
        return chunk.detach().clone()

    def select_action(self, observation: Mapping[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        """Return the next action, refilling the queue only when it is empty.

        Args:
            observation: Raw observations, checked even while a cached chunk is consumed.

        Returns:
            An owned ``[batch, action]`` tensor.
        """
        _, count, _ = self._action_contract()
        if not self._actions:
            chunk = self.predict_action_chunk(observation)
            self._actions.extend(chunk[:, :count].unbind(dim=1))
        else:
            self._prepare_inputs(observation)
        return self._actions.popleft().clone()

    def reset(self) -> None:
        """Discard queued actions and restore the runtime's initial buffers."""
        self._actions.clear()
        self.manager.reset()
