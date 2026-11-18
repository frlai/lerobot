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

"""Capture existing deterministic numerical callables as one portable LEAPP node."""

import re
import shutil
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from lerobot.utils.import_utils import _leapp_available, require_package

from .annotations import port_aliases, resolve_semantics

if TYPE_CHECKING or _leapp_available:
    import leapp

type TensorMap = Mapping[str, torch.Tensor | np.ndarray]


def _copy_values(values: TensorMap) -> dict[str, Any]:
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value.copy()
        for key, value in values.items()
    }


def _validate_values(values: TensorMap, direction: str) -> None:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"Pipeline {direction} must be a nonempty mapping of named arrays/tensors")
    port_aliases(values)
    for key, value in values.items():
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise TypeError(f"Pipeline {direction} {key!r} must be an array or tensor")


def export_pipeline(
    pipeline: Callable[[dict[str, Any]], TensorMap],
    samples: Iterable[TensorMap],
    output_dir: str | Path,
    *,
    name: str = "policy",
    backend: str = "onnx",
) -> Path:
    """Export one fixed-shape, deterministic numerical pipeline and validate its samples.

    ``pipeline`` runs unchanged on independent copies of each sample. Include all
    numeric pre/postprocessing inside that callable. Samples must share input keys,
    shapes and dtypes, and must exercise the configuration intended for deployment.
    Python data-dependent branches and arbitrary mutable state are not qualified.

    The returned YAML and every file in its directory form the deployment bundle.
    Existing bundles are never overwritten. Export errors restore LEAPP patches and
    remove the incomplete directory created by this call.
    """
    require_package("leapp", "leapp")
    if leapp.__version__ != "0.8.1":
        raise RuntimeError(f"This integration requires leapp==0.8.1, found {leapp.__version__}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Export name must be a Python identifier")
    if backend not in {"onnx", "onnx-dynamo", "onnx-torchscript", "pt2", "jit-trace"}:
        raise ValueError(f"Unsupported export backend: {backend!r}")
    if leapp.ExportManager.is_interpret_graph_enabled():
        raise RuntimeError("Cannot export inside an active LEAPP capture session")
    batches = list(samples)
    if not batches:
        raise ValueError("At least one capture sample is required")
    for batch in batches:
        _validate_values(batch, "inputs")
    first = batches[0]
    signature = {key: (tuple(value.shape), str(value.dtype)) for key, value in first.items()}
    for batch in batches[1:]:
        if {key: (tuple(value.shape), str(value.dtype)) for key, value in batch.items()} != signature:
            raise ValueError("Capture samples must have identical input keys, shapes and dtypes")
    input_aliases = port_aliases(first)
    bundle_dir = Path(output_dir).resolve() / name
    if bundle_dir.exists():
        raise FileExistsError(f"Export directory already exists: {bundle_dir}")
    deployment = deepcopy(getattr(pipeline, "_leapp_deployment", {}))
    output_aliases: dict[str, str] | None = None
    output_signature = None
    try:
        leapp.start(name=name, save_path=str(bundle_dir.parent), max_cached_io=len(batches))
        try:
            with torch.inference_mode():
                for batch in batches:
                    values = _copy_values({key: batch[key] for key in first})
                    traced = leapp.annotate.input_tensors(
                        name, resolve_semantics(pipeline, "inputs", values, input_aliases)
                    )
                    traced_values = (traced,) if len(values) == 1 else traced
                    outputs = pipeline(dict(zip(values, traced_values, strict=True)))
                    _validate_values(outputs, "outputs")
                    current_signature = {
                        key: (tuple(value.shape), str(value.dtype)) for key, value in outputs.items()
                    }
                    if output_aliases is None:
                        output_aliases = port_aliases(outputs)
                        output_signature = current_signature
                    elif current_signature != output_signature:
                        raise ValueError("Pipeline output keys, shapes and dtypes changed across samples")
                    outputs = {key: outputs[key] for key in output_aliases}
                    leapp.annotate.output_tensors(
                        name,
                        resolve_semantics(pipeline, "outputs", outputs, output_aliases),
                        export_with=backend,
                    )
        finally:
            leapp.stop()
        metadata = {
            **deployment,
            "schema_version": 1,
            "input_aliases": {key: f"{name}/{alias}" for key, alias in input_aliases.items()},
            "output_aliases": {key: f"{name}/{alias}" for key, alias in (output_aliases or {}).items()},
        }
        metadata.setdefault("provenance", {})["leapp_version"] = leapp.__version__
        leapp.compile_graph(
            visualize=False,
            validate=True,
            strict=True,
            rtol=1e-4,
            atol=1e-4,
            graph_configs=leapp.GraphConfigs(extra={"lerobot": metadata}),
        )
        bundle = bundle_dir / f"{name}.yaml"
        description = yaml.safe_load(bundle.read_text())
        live_inputs = {
            f"{node}/{port}" for node, ports in description["pipeline"]["inputs"].items() for port in ports
        }
        if live_inputs != set(metadata["input_aliases"].values()):
            raise ValueError("Some declared observations are unused; remove unused keys from capture samples")
        return bundle
    except BaseException:
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        raise
