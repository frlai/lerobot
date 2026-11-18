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

import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from lerobot.export.annotations import export_semantics, port_aliases, resolve_semantics


def test_decorator_preserves_callable_and_defers_resolution():
    calls = []

    def resolver(value):
        calls.append(value)
        return {"axes": [None] * value.ndim}

    def pipeline(observation):
        return {"action": observation["state"] * 2}

    original = pipeline
    pipeline = export_semantics(outputs={"action": resolver})(pipeline)
    assert pipeline is original
    observation = {"state": torch.tensor([1.0, 2.0])}
    torch.testing.assert_close(pipeline(observation)["action"], torch.tensor([2.0, 4.0]))
    assert calls == []


def test_annotations_import_and_decoration_without_leapp():
    script = """
import builtins
import sys

original_import = builtins.__import__

def import_without_leapp(name, *args, **kwargs):
    if name == "leapp" or name.startswith("leapp."):
        raise AssertionError("annotations attempted to import LEAPP")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_leapp
from lerobot.export.annotations import export_semantics, port_aliases
from lerobot.processor import PolicyProcessorPipeline
from lerobot.policies.act.modeling_act import ACTPolicy

@export_semantics(outputs={"action": {"axes": [None]}})
def pipeline(x):
    return x + 1

assert pipeline(2) == 3
assert port_aliases(["observation.state"]) == {"observation.state": "observation_state"}
assert "leapp" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_port_aliases_preserve_external_names_and_order():
    names = ["observation.images.front", "observation.state", "1.action", "class", "self"]
    aliases = port_aliases(names)
    assert list(aliases) == names
    assert aliases == {
        "observation.images.front": "observation_images_front",
        "observation.state": "observation_state",
        "1.action": "port_1_action",
        "class": "port_class",
        "self": "port_self",
    }
    assert all(alias.isidentifier() for alias in aliases.values())


@pytest.mark.parametrize(
    "names,match",
    [
        (["observation.state", "observation_state"], "both map to"),
        (["class", "port_class"], "both map to"),
        (["action", "action"], "Duplicate"),
        ([""], "nonempty"),
        ([" "], "nonempty"),
        ([None], "nonempty"),
    ],
)
def test_port_aliases_reject_ambiguous_or_invalid_names(names, match):
    with pytest.raises(ValueError, match=match):
        port_aliases(names)


def test_resolve_semantics_binds_values_and_serializes_layouts():
    leapp = pytest.importorskip("leapp")
    resolved_values = []

    def action_metadata(value):
        resolved_values.append(value)
        return {"axes": [None, {"kind": "time", "period_ms": 100}, {"kind": "element", "names": ["x", "y"]}]}

    @export_semantics(
        inputs={"observation.image": {"kind": "image", "axes": [None, None, None, None]}},
        outputs={"action": action_metadata},
    )
    def pipeline(observation):
        return observation

    inputs = {
        "observation.image": np.zeros((1, 4, 5, 3), dtype=np.uint8),
        "observation.state": torch.ones(1, 2),
    }
    outputs = {"action": torch.zeros(1, 3, 2)}
    input_metadata = resolve_semantics(pipeline, "inputs", inputs, port_aliases(inputs))
    output_metadata = resolve_semantics(pipeline, "outputs", outputs, port_aliases(outputs))

    assert all(isinstance(semantics, leapp.TensorSemantics) for semantics in input_metadata + output_metadata)
    assert [semantics.name for semantics in input_metadata] == ["observation_image", "observation_state"]
    assert input_metadata[0].ref is inputs["observation.image"]
    assert input_metadata[1].ref is inputs["observation.state"]
    assert input_metadata[1].to_dict() == {}
    assert resolved_values == [outputs["action"]]
    serialized = yaml.safe_load(yaml.safe_dump(output_metadata[0].to_dict()))
    assert serialized == {
        "axes": [None, {"kind": "time", "period_ms": 100}, {"kind": "element", "names": ["x", "y"]}]
    }
    assert "kind" not in serialized  # Ordered names do not establish a physical quantity.
    assert "spatial" not in serialized


@pytest.mark.parametrize(
    "metadata,match",
    [
        ({"axes": [None]}, "one entry per tensor dimension"),
        ({"axes": [None, {"kind": "element", "names": ["x"]}]}, "label count"),
        ({"axes": [None, {"kind": "time"}]}, "requires period_ms"),
        ({"axes": [{"kind": "time", "period_ms": 1}, {"kind": "time", "period_ms": 1}]}, "at most one"),
        ({"is_setpoint": True}, "only to inputs"),
        ({"ref": None}, "must not override"),
        ({"name": "other"}, "must not override"),
    ],
)
def test_resolve_semantics_rejects_invalid_output_metadata(metadata, match):
    pytest.importorskip("leapp")

    @export_semantics(outputs={"action": metadata})
    def pipeline(observation):
        return observation

    values = {"action": torch.zeros(1, 2)}
    with pytest.raises(ValueError, match=match):
        resolve_semantics(pipeline, "outputs", values, port_aliases(values))


def test_resolve_semantics_rejects_unknown_ports():
    pytest.importorskip("leapp")

    @export_semantics(inputs={"missing": {"kind": "image"}})
    def pipeline(observation):
        return observation

    with pytest.raises(ValueError, match="unknown inputs.*missing"):
        resolve_semantics(pipeline, "inputs", {"state": torch.ones(2)}, {"state": "state"})
