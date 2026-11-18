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

"""Evaluate an exported LEAPP action-chunk bundle in PushT, without policy source imports."""

import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import gymnasium as gym
import numpy as np

from lerobot.export.runtime import LeappRunner
from lerobot.utils.import_utils import require_package


@dataclass
class PushTEvalConfig:
    """PushT settings matching LeRobot's default image/state environment."""

    task: str = "gym_pusht/PushT-v0"
    episode_length: int = 300
    fps: int = 10


@dataclass
class LeappEvalConfig:
    """Configuration for ``lerobot-eval-leapp``."""

    bundle_path: Path
    output_dir: Path = Path("outputs/eval/leapp")
    n_episodes: int = 50
    start_seed: int = 0
    env: PushTEvalConfig = field(default_factory=PushTEvalConfig)


def pusht_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Name and batch raw PushT observations; all numerical preprocessing stays in the model."""
    return {
        "observation.image": observation["pixels"][None],
        "observation.state": np.asarray(observation["agent_pos"], dtype=np.float32)[None],
    }


def evaluate(cfg: LeappEvalConfig) -> dict[str, Any]:
    """Run complete seeded episodes using only the deployment bundle.

    Args:
        cfg: Bundle, output path, and PushT evaluation settings.

    Returns:
        Per-episode rewards and successes, together with their aggregate values.
    """
    if cfg.n_episodes <= 0 or cfg.env.episode_length <= 0 or cfg.env.fps <= 0:
        raise ValueError("Episode count, episode length, and FPS must be positive.")
    if cfg.env.task != "gym_pusht/PushT-v0":
        raise ValueError("lerobot-eval-leapp currently supports gym_pusht/PushT-v0 only.")
    require_package("gym-pusht", extra="pusht", import_name="gym_pusht")
    importlib.import_module("gym_pusht")  # Registers the optional Gymnasium environment.
    runner = LeappRunner(cfg.bundle_path)
    if not math.isclose(runner.config.get("control_fps", float("nan")), cfg.env.fps):
        raise ValueError("Environment FPS must match the bundle's control_fps.")
    env = gym.make(
        cfg.env.task,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=cfg.env.episode_length,
    )
    episodes = []
    try:
        for seed in range(cfg.start_seed, cfg.start_seed + cfg.n_episodes):
            runner.reset()
            observation, _ = env.reset(seed=seed)
            reward_sum, max_reward, success = 0.0, 0.0, False
            steps = 0
            for _ in range(cfg.env.episode_length):
                steps += 1
                action_batch = runner.select_action(pusht_observation(observation))
                if action_batch.shape[0] != 1 or tuple(action_batch.shape[1:]) != env.action_space.shape:
                    raise ValueError("Exported actions must match PushT's action shape with batch size one.")
                action = action_batch.cpu().numpy()[0]
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                max_reward = max(max_reward, float(reward))
                success = success or bool(info.get("is_success", terminated))
                if terminated or truncated:
                    break
            episodes.append(
                {
                    "seed": seed,
                    "reward_sum": reward_sum,
                    "max_reward": max_reward,
                    "success": success,
                    "steps": steps,
                }
            )
    finally:
        env.close()
    result = {
        "bundle_path": str(cfg.bundle_path),
        "n_episodes": cfg.n_episodes,
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "mean_reward_sum": float(np.mean([episode["reward_sum"] for episode in episodes])),
        "mean_max_reward": float(np.mean([episode["max_reward"] for episode in episodes])),
        "episodes": episodes,
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "eval_info.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@draccus.wrap()
def eval_main(cfg: LeappEvalConfig) -> None:
    """Parse configuration and report the exported policy's PushT metrics."""
    result = evaluate(cfg)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


def main() -> None:
    """Run the command-line entry point."""
    eval_main()


if __name__ == "__main__":
    main()
