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

"""Qualify a real ACT checkpoint through export, scheduling and seeded PushT deployment.

Run with the leapp, pusht and evaluation extras. Assets and results stay outside
source control. No training or hardware is needed; every comparison runs on CPU.
"""

import argparse
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from lerobot.envs.utils import prepare_canonical_observation
from lerobot.export.act import DEFAULT_ACT_CHECKPOINT, DEFAULT_ACT_REVISION, load_act_pipeline
from lerobot.export.runtime import LeappRunner
from lerobot.scripts.lerobot_eval_leapp import LeappEvalConfig, evaluate, pusht_observation
from lerobot.scripts.lerobot_export_leapp import LeappExportConfig, export_main


def qualify(checkpoint_path: str, revision: str, output_dir: Path, n_episodes: int) -> dict:
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    torch.set_num_threads(1)
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = export_main(
        LeappExportConfig(
            checkpoint_path=checkpoint_path, revision=revision, output_dir=output_dir / "export"
        )
    )
    description = yaml.safe_load(bundle.read_text())
    assert len(description["models"]) == 1
    assert description["models"]["policy"]["parameters"]["backend"] == "onnx"
    pipeline = load_act_pipeline(checkpoint_path, revision)
    runner = LeappRunner(bundle)
    importlib.import_module("gym_pusht")
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=300
    )
    samples = []
    max_error = 0.0
    comparison_count = 0

    def compare(actual, expected):
        nonlocal max_error, comparison_count
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        max_error = max(max_error, (actual - expected).abs().max().item())
        comparison_count += 1

    try:
        # Different seeds from the five capture samples; independent image/state perturbations.
        for seed in range(100, 120):
            raw, _ = env.reset(seed=seed)
            observation = {key: value.copy() for key, value in pusht_observation(raw).items()}
            if seed % 3 == 0:
                image = observation["observation.image"]
                observation["observation.image"] = np.clip(image.astype(np.int16) + 7, 0, 255).astype(
                    np.uint8
                )
            elif seed % 3 == 1:
                observation["observation.state"] += np.array([[5.0, -3.0]], dtype=np.float32)
            compare(runner.predict_action_chunk(observation), pipeline(observation)["action_chunk"])
            samples.append(observation)

        saved_observation = {key: value.copy() for key, value in samples[0].items()}
        owned_chunk = runner.predict_action_chunk(samples[0])
        saved_chunk = owned_chunk.clone()
        owned_action = runner.select_action(samples[0])
        saved_action = owned_action.clone()
        runner.predict_action_chunk(samples[1])
        runner.reset()
        for key, expected in saved_observation.items():
            np.testing.assert_array_equal(samples[0][key], expected)
        torch.testing.assert_close(owned_chunk, saved_chunk, rtol=0, atol=0)
        torch.testing.assert_close(owned_action, saved_action, rtol=0, atol=0)

        # A whole-chunk postprocessor must agree with the original per-action call.
        batch = pipeline.preprocessor(prepare_canonical_observation(samples[0]))
        chunk = pipeline.policy.predict_action_chunk(batch)
        per_action = torch.stack([pipeline.postprocessor(chunk[:, i]) for i in range(chunk.shape[1])], dim=1)
        torch.testing.assert_close(pipeline.postprocessor(chunk), per_action, rtol=0, atol=0)

        # At least three queue refills, and reset while a chunk remains queued.
        schedule_steps = 0
        for steps in (
            3 * pipeline.policy.config.n_action_steps + 1,
            pipeline.policy.config.n_action_steps + 1,
        ):
            runner.reset()
            pipeline.policy.reset()
            pipeline.preprocessor.reset()
            pipeline.postprocessor.reset()
            for step in range(steps):
                observation = samples[step % len(samples)]
                native = pipeline.postprocessor(
                    pipeline.policy.select_action(
                        pipeline.preprocessor(prepare_canonical_observation(observation))
                    )
                )
                compare(runner.select_action(observation), native)
                schedule_steps += 1

        # Replay a relocated real bundle in a process that cannot import source policies/processors.
        relocated = output_dir / "relocated"
        shutil.copytree(bundle.parent, relocated)
        fixture = output_dir / "observation.npz"
        np.savez(fixture, image=samples[0]["observation.image"], state=samples[0]["observation.state"])
        output = output_dir / "relocated_output.npy"
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.abc,sys\n"
                "class BlockSources(importlib.abc.MetaPathFinder):\n"
                " def find_spec(self,fullname,path=None,target=None):\n"
                "  if fullname.startswith(('lerobot.policies','lerobot.processor')): raise ImportError(fullname)\n"
                "sys.meta_path.insert(0,BlockSources())\n"
                "import numpy as np\n"
                "from lerobot.export.runtime import LeappRunner\n"
                "data=np.load(sys.argv[2])\n"
                "obs={'observation.image':data['image'],'observation.state':data['state']}\n"
                "np.save(sys.argv[3],LeappRunner(sys.argv[1]).predict_action_chunk(obs).numpy())\n",
                str(relocated / bundle.name),
                str(fixture),
                str(output),
            ],
            check=True,
            timeout=120,
        )
        compare(torch.from_numpy(np.load(output)), pipeline(samples[0])["action_chunk"])

        # Native closed-loop baseline. Shadow the exported chunk on each shared refill observation.
        native_episodes = []
        rollout_comparisons = 0
        for seed in range(n_episodes):
            pipeline.policy.reset()
            pipeline.preprocessor.reset()
            pipeline.postprocessor.reset()
            raw, _ = env.reset(seed=seed)
            reward_sum, max_reward, success = 0.0, 0.0, False
            for step in range(300):
                observation = pusht_observation(raw)
                if step % pipeline.policy.config.n_action_steps == 0:
                    compare(runner.predict_action_chunk(observation), pipeline(observation)["action_chunk"])
                    rollout_comparisons += 1
                action = pipeline.postprocessor(
                    pipeline.policy.select_action(
                        pipeline.preprocessor(prepare_canonical_observation(observation))
                    )
                )
                raw, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
                reward_sum += float(reward)
                max_reward = max(max_reward, float(reward))
                success = success or bool(info.get("is_success", terminated))
                if terminated or truncated:
                    break
            native_episodes.append(
                {
                    "seed": seed,
                    "steps": step + 1,
                    "reward_sum": reward_sum,
                    "max_reward": max_reward,
                    "success": success,
                }
            )
            print(f"Native episode {seed + 1}/{n_episodes}", flush=True)
    finally:
        env.close()

    deployed = evaluate(
        LeappEvalConfig(bundle_path=bundle, output_dir=output_dir / "deployment", n_episodes=n_episodes)
    )
    native = {
        "success_rate": float(np.mean([episode["success"] for episode in native_episodes])),
        "mean_reward_sum": float(np.mean([episode["reward_sum"] for episode in native_episodes])),
        "mean_max_reward": float(np.mean([episode["max_reward"] for episode in native_episodes])),
        "episodes": native_episodes,
    }
    report = {
        "checkpoint": pipeline.provenance,
        "torch_version": torch.__version__,
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "lerobot",
                "leapp",
                "torchvision",
                "numpy",
                "onnx",
                "onnxscript",
                "onnxruntime",
                "gym-pusht",
            )
        },
        "bundle": str(bundle),
        "capture_samples": 5,
        "fresh_samples": len(samples),
        "schedule_steps": schedule_steps,
        "shared_rollout_chunk_comparisons": rollout_comparisons,
        "numerical_comparisons": comparison_count,
        "max_absolute_error": max_error,
        "rtol": 1e-4,
        "atol": 1e-4,
        "inplace_relu_count": sum(
            isinstance(module, torch.nn.ReLU) and module.inplace for module in pipeline.policy.modules()
        ),
        "relocated_policy_free_replay": True,
        "caller_buffer_ownership": True,
        "native": native,
        "deployed": deployed,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default=DEFAULT_ACT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_ACT_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    args = parser.parse_args()
    result = qualify(args.checkpoint_path, args.revision, args.output_dir.resolve(), args.n_episodes)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("native", "deployed")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
