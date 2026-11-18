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

"""Deferred metadata for external pipeline ports, using LEAPP's semantic vocabulary."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from keyword import iskeyword
from typing import TYPE_CHECKING, Any, Literal

from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    from leapp import TensorSemantics

type SemanticFields = Mapping[str, Any]
type SemanticDeclaration = SemanticFields | Callable[[Any], SemanticFields]


def export_semantics[F: Callable[..., Any]](
    *,
    inputs: Mapping[str, SemanticDeclaration] | None = None,
    outputs: Mapping[str, SemanticDeclaration] | None = None,
) -> Callable[[F], F]:
    """Attach external-port semantics without importing LEAPP or wrapping the callable.

    Keys are the pipeline's original input/output names. Values contain LEAPP
    ``TensorSemantics`` fields, with ``axes`` expressed as dictionaries or ``None``.
    A callable value defers those fields until export and receives the actual tensor.
    Resolvers must only inspect metadata, such as its shape; they must not mutate
    values or derive a contract from the capture sample's numerical contents.
    ``name`` and ``ref`` are supplied by the exporter and must not be repeated.

    For example, ``outputs={"actions": {"axes": [None, {"kind": "time",
    "period_ms": 100}, {"kind": "element", "names": ["x", "y"]}]}}``
    describes a batched action chunk without guessing its physical quantity.
    """
    metadata = {"inputs": deepcopy(dict(inputs or {})), "outputs": deepcopy(dict(outputs or {}))}

    def decorate(pipeline: F) -> F:
        pipeline._leapp_semantics = metadata  # type: ignore[attr-defined]
        return pipeline

    return decorate


def port_aliases(names: Iterable[str]) -> dict[str, str]:
    """Map external names to Python-safe LEAPP ports, rejecting ambiguous aliases."""
    aliases: dict[str, str] = {}
    external_by_alias: dict[str, str] = {}
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Pipeline port names must be nonempty strings")
        if name in aliases:
            raise ValueError(f"Duplicate pipeline port name: {name!r}")
        alias = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if alias[0].isdigit() or iskeyword(alias) or alias == "self":
            alias = f"port_{alias}"
        if alias in external_by_alias:
            raise ValueError(
                f"Pipeline ports {external_by_alias[alias]!r} and {name!r} both map to {alias!r}; "
                "rename one external port"
            )
        aliases[name] = alias
        external_by_alias[alias] = name
    return aliases


def resolve_semantics(
    pipeline: Callable[..., Any],
    direction: Literal["inputs", "outputs"],
    values: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> list[TensorSemantics]:
    """Bind deferred metadata to actual values using released LEAPP descriptors.

    The returned homogeneous list includes unannotated values with empty semantics,
    preserving the order in ``values`` and avoiding mixed LEAPP/raw payloads.
    """
    require_package("leapp", "leapp")
    # Resolution is the optional export boundary; ordinary annotations never import LEAPP.
    from leapp import Axis, TensorSemantics

    if direction not in {"inputs", "outputs"}:
        raise ValueError(f"Unknown semantic direction: {direction!r}")
    declarations = getattr(pipeline, "_leapp_semantics", {}).get(direction, {})
    unknown = declarations.keys() - values.keys()
    if unknown:
        raise ValueError(f"Semantics refer to unknown {direction}: {sorted(unknown)}")
    semantics = []
    for name, value in values.items():
        declaration = declarations.get(name, {})
        fields = dict(declaration(value) if callable(declaration) else declaration)
        if fields.keys() & {"name", "ref"}:
            raise ValueError("Semantic fields must not override the external port's name or value")
        axes = fields.get("axes")
        if axes is not None:
            fields["axes"] = [Axis(**axis) if isinstance(axis, Mapping) else axis for axis in axes]
        if direction == "outputs" and fields.get("is_setpoint"):
            raise ValueError("is_setpoint=True applies only to inputs")
        semantics.append(TensorSemantics(name=aliases[name], ref=value, **fields))
    return semantics
