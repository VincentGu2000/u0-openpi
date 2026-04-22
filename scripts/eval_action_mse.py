"""Evaluate action prediction MSE on a test dataset (per-trajectory).

Computes MSE between model-predicted actions and ground truth actions from a
LeRobot test dataset. Supports optional visualization via --plot or --save_plot_path.
See README-U0.md for detailed usage.
"""

# Monkey-patch to fix 'List' feature type error in old datasets
try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
except (ImportError, AttributeError):
    pass
# End of monkey-patch

import argparse
import csv
import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.policies import policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

logger = logging.getLogger("openpi")


def to_numpy(data):
    """Convert torch tensors or other array-like objects to numpy arrays."""
    if isinstance(data, torch.Tensor):
        return data.numpy()
    return np.asarray(data)


def load_norm_stats_from_dataset(repo_id: str):
    """Load norm_stats from a LeRobot dataset directory.

    Due to pathlib behavior (absolute path wins in Path(a) / "/absolute/b"),
    compute_norm_stats.py saves norm_stats directly inside the dataset directory
    when repo_id is an absolute path.
    """
    return _normalize.load(repo_id)


def get_episode_info(dataset):
    """Extract per-episode frame index ranges from a LeRobot dataset.

    Uses the dataset's ``episode_data_index`` attribute (a dict with ``from`` and ``to``
    keys) which maps each episode index to its start/end frame indices in the dataset.

    Returns:
        episode_indices: sorted list of unique episode indices
        episode_frame_ranges: dict mapping episode_index -> (start_frame_idx, end_frame_idx) in the dataset
    """
    from_idx = to_numpy(dataset.episode_data_index["from"])
    to_idx = to_numpy(dataset.episode_data_index["to"])
    num_episodes = dataset.num_episodes

    episode_indices = list(range(num_episodes))
    episode_frame_ranges = {}
    for ep_idx in episode_indices:
        episode_frame_ranges[ep_idx] = (int(from_idx[ep_idx]), int(to_idx[ep_idx]) - 1)

    return episode_indices, episode_frame_ranges


def plot_trajectory(gt_actions, pred_actions, states, traj_id, mse,
                    action_horizon, show=True, save_path=None):
    """Plot per-dimension comparison of ground truth vs predicted actions.

    Args:
        gt_actions: np.ndarray of shape (steps, action_dim)
        pred_actions: np.ndarray of shape (steps, action_dim)
        states: np.ndarray of shape (steps, state_dim) or empty array
        traj_id: episode index for the title
        mse: MSE value for the title
        action_horizon: interval for marking inference points
        show: whether to show interactive matplotlib window
        save_path: file path to save the plot (None = don't save)
    """
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_dim = gt_actions.shape[1]
    steps = gt_actions.shape[0]

    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    fig.suptitle(
        f"Trajectory {traj_id} - Action MSE: {mse:.6f}\n"
        f"Steps: {steps}, Action Horizon: {action_horizon}",
        fontsize=14, fontweight="bold", color="#2E86AB", y=0.95,
    )

    for i, ax in enumerate(axes):
        # Plot state if dimensions match
        if len(states) > 0 and states.shape[1] == action_dim:
            ax.plot(states[:, i], label="state", alpha=0.7)
        ax.plot(gt_actions[:, i], label="gt action", linewidth=2)
        ax.plot(pred_actions[:, i], label="pred action", linewidth=2)

        # Mark inference points every action_horizon
        for j in range(0, steps, action_horizon):
            ax.plot(j, gt_actions[j, i], "ro", markersize=4)

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"  Saving plot to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def calc_mse_for_single_trajectory(
    policy,
    dataset,
    episode_idx,
    frame_range,
    action_horizon,
    fps,
    eval_horizon=None,
    plot=False,
    save_plot_path=None,
):
    """Compute action MSE for a single trajectory

    Returns:
        mse: global MSE across all valid step-level pairs in the trajectory
        n_steps: number of step-level pairs evaluated
    """
    if eval_horizon is None:
        eval_horizon = action_horizon
    assert eval_horizon <= action_horizon, (
        f"eval_horizon ({eval_horizon}) must be <= action_horizon ({action_horizon})"
    )

    start_idx, end_idx = frame_range
    # Avoid evaluating the last `action_horizon - 1` frames where the GT action
    # chunk would extend beyond the episode boundary (and thus be NaN).
    traj_length = end_idx - start_idx + 1
    steps = max(traj_length - action_horizon, 0)

    if steps == 0:
        return None, 0

    gt_action_across_time = []
    pred_action_across_time = []
    states_across_time = []

    for step_count in range(steps):
        data_idx = start_idx + step_count
        sample = dataset[data_idx]

        # Collect state for visualization at every step
        if plot or save_plot_path is not None:
            states_across_time.append(to_numpy(sample["observation.state"]))

        # Only run inference at eval_horizon intervals (U0 protocol)
        if step_count % eval_horizon == 0:
            # Get prompt: prefer the "task" field directly, fall back to task_index lookup
            if "task" in sample and isinstance(sample["task"], str):
                prompt = sample["task"]
            else:
                task_index = int(to_numpy(sample["task_index"]))
                prompt = dataset.meta.tasks.get(task_index, "do something")

            # Construct observation dict (repack from LeRobot format to policy input format)
            obs = {
                "observation/ego_image": to_numpy(sample["observation.images.ego"]),
                "observation/wrist_image": to_numpy(sample["observation.images.wrist"]),
                "observation/state": to_numpy(sample["observation.state"]),
                "prompt": prompt,
            }

            # Ground truth action chunk: (eval_horizon, action_dim)
            gt_chunk = to_numpy(sample["action"])

            # Skip this inference point if ground truth has NaN
            if np.any(np.isnan(gt_chunk)):
                continue

            # Run inference
            result = policy.infer(obs)
            pred_chunk = result["actions"]  # (action_horizon, 13)

            # Clip predictions to [-1, 1] range
            pred_chunk = np.clip(pred_chunk, -1.0, 1.0)

            # Unfold the chunk: collect step-level pred/gt pairs (limited by eval_horizon)
            max_j = min(gt_chunk.shape[0], pred_chunk.shape[0], eval_horizon)
            action_dim = min(gt_chunk.shape[1], pred_chunk.shape[1])
            for j in range(max_j):
                # Only collect if within valid steps range
                if step_count + j < steps:
                    pred_action_across_time.append(pred_chunk[j, :action_dim])
                    gt_action_across_time.append(gt_chunk[j, :action_dim])

    if len(gt_action_across_time) == 0:
        return None, 0

    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)

    # Compute global MSE over all step-level pairs (U0 protocol)
    assert gt_action_across_time.shape == pred_action_across_time.shape
    mse = float(np.mean((gt_action_across_time - pred_action_across_time) ** 2))

    # Generate visualization if requested
    if plot or save_plot_path is not None:
        states_across_time = np.array(states_across_time)[:steps]

        actual_save_path = None
        if save_plot_path is not None:
            actual_save_path = f"{save_plot_path}/traj_{episode_idx}.png"

        plot_trajectory(
            gt_actions=gt_action_across_time,
            pred_actions=pred_action_across_time,
            states=states_across_time,
            traj_id=episode_idx,
            mse=mse,
            action_horizon=eval_horizon,
            show=plot,
            save_path=actual_save_path,
        )

    return mse, len(gt_action_across_time)


def main():
    parser = argparse.ArgumentParser(description="Evaluate action prediction MSE on a test dataset (per-trajectory)")
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_u0bot",
        help="Training config name (default: pi05_u0bot)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the trained model checkpoint directory",
    )
    parser.add_argument(
        "--test_repo_id",
        type=str,
        default="/data/gujunwen/project/fish-vla/dataset/lerobot_test",
        help="LeRobot dataset repo ID or local path for testing",
    )
    parser.add_argument(
        "--train_repo_id",
        type=str,
        default=None,
        help=(
            "Training dataset path (for loading norm_stats). "
            "If not provided, uses the repo_id from the training config."
        ),
    )
    parser.add_argument(
        "--save_csv_path",
        type=str,
        default=None,
        help="Path to save per-trajectory results CSV (e.g., results/eval.csv)",
    )
    parser.add_argument(
        "--max_trajs",
        type=int,
        default=None,
        help="Maximum number of trajectories to evaluate (None = all)",
    )
    parser.add_argument(
        "--start_traj",
        type=int,
        default=0,
        help="Start trajectory index (default: 0)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show interactive matplotlib plots for each trajectory",
    )
    parser.add_argument(
        "--save_plot_path",
        type=str,
        default=None,
        help="Directory to save per-trajectory plots (e.g., results/plots)",
    )
    parser.add_argument(
        "--eval_horizon",
        type=int,
        default=None,
        help=(
            "Number of steps to compare from each predicted chunk (default: full action_horizon). "
            "Use 1 to only compare the first step of each chunk (single-step prediction quality). "
            "Must be <= action_horizon."
        ),
    )
    args = parser.parse_args()

    # Create save_plot_path directory if specified
    if args.save_plot_path is not None:
        Path(args.save_plot_path).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load config and create trained policy
    # ----------------------------------------------------------------
    print(f"[1/3] Loading config: {args.config_name}")
    config = _config.get_config(args.config_name)
    action_horizon = config.model.action_horizon
    print(f"  Action horizon: {action_horizon}")

    # Load norm_stats from the training dataset directory
    train_repo_id = args.train_repo_id
    if train_repo_id is None:
        if hasattr(config.data, "repo_id"):
            train_repo_id = config.data.repo_id
        else:
            raise ValueError("Cannot determine training repo_id. Please provide --train_repo_id.")

    print(f"  Loading norm_stats from training dataset: {train_repo_id}")
    norm_stats = load_norm_stats_from_dataset(train_repo_id)
    print(f"  Norm stats keys: {list(norm_stats.keys())}")

    print(f"  Loading model from: {args.checkpoint_dir}")
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        norm_stats=norm_stats,
    )
    print("  Model loaded successfully.")

    # ----------------------------------------------------------------
    # 2. Load test LeRobot dataset with action chunks
    # ----------------------------------------------------------------
    print(f"\n[2/3] Loading test dataset: {args.test_repo_id}")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(args.test_repo_id)
    fps = dataset_meta.fps
    print(f"  FPS: {fps}")
    print(f"  Total episodes: {dataset_meta.total_episodes}")
    print(f"  Tasks ({len(dataset_meta.tasks)}): {dataset_meta.tasks}")

    dataset = lerobot_dataset.LeRobotDataset(
        args.test_repo_id,
        delta_timestamps={
            "action": [t / fps for t in range(action_horizon)],
        },
    )
    print(f"  Dataset size: {len(dataset)} samples")

    # Get episode information
    episode_indices, episode_frame_ranges = get_episode_info(dataset)
    total_trajs = len(episode_indices)
    print(f"  Unique episodes: {total_trajs}")

    # Determine the range of trajectories to evaluate
    start_idx = args.start_traj
    end_idx = total_trajs if args.max_trajs is None else min(start_idx + args.max_trajs, total_trajs)
    print(f"  Evaluating trajectories {start_idx} to {end_idx - 1}")

    # ----------------------------------------------------------------
    # 3. Evaluate per-trajectory
    # ----------------------------------------------------------------
    print(f"\n[3/3] Running evaluation...")
    all_mse = []
    all_traj_steps = []
    all_traj_lengths = []

    # Open CSV file for real-time writing
    csv_file = None
    csv_writer = None
    if args.save_csv_path is not None:
        csv_path = Path(args.save_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["traj_id", "traj_length", "eval_steps", "action_mse"])
        csv_file.flush()

    start_time = time.time()

    for idx in tqdm.tqdm(range(start_idx, end_idx), desc="Evaluating trajectories"):
        ep_idx = episode_indices[idx]
        frame_range = episode_frame_ranges[ep_idx]
        traj_length = frame_range[1] - frame_range[0] + 1

        mse, n_steps = calc_mse_for_single_trajectory(
            policy,
            dataset,
            ep_idx,
            frame_range,
            action_horizon,
            fps,
            eval_horizon=args.eval_horizon,
            plot=args.plot,
            save_plot_path=args.save_plot_path,
        )

        if mse is None or n_steps == 0:
            print(f"  Skipping trajectory {ep_idx} (length={traj_length}, too short)")
            continue

        print(
            f"  [{idx - start_idx + 1}/{end_idx - start_idx}] "
            f"Trajectory {ep_idx}: length={traj_length}, steps={n_steps}, MSE={mse:.6f}"
        )

        all_mse.append(mse)
        all_traj_steps.append(n_steps)
        all_traj_lengths.append(traj_length)

        # Write to CSV immediately
        if csv_writer is not None:
            csv_writer.writerow([ep_idx, traj_length, n_steps, f"{mse:.6f}"])
            csv_file.flush()

        # Release memory
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    elapsed = time.time() - start_time

    # ----------------------------------------------------------------
    # 4. Report results
    # ----------------------------------------------------------------
    if len(all_mse) == 0:
        print("\nNo trajectories were evaluated!")
        if csv_file is not None:
            csv_file.close()
        return

    all_mse_arr = np.array(all_mse)
    all_traj_steps_arr = np.array(all_traj_steps, dtype=np.float64)

    simple_avg_mse = np.mean(all_mse_arr)
    simple_std_mse = np.std(all_mse_arr)
    weighted_mse = np.sum(all_mse_arr * all_traj_steps_arr) / np.sum(all_traj_steps_arr)

    print(f"\n{'=' * 60}")
    print(f"  Evaluation Results - Action MSE")
    print(f"{'=' * 60}")
    print(f"  Config:             {args.config_name}")
    print(f"  Checkpoint:         {args.checkpoint_dir}")
    print(f"  Test dataset:       {args.test_repo_id}")
    print(f"  Evaluated trajs:    {len(all_mse)}")
    print(f"  Total eval steps:   {int(np.sum(all_traj_steps_arr))}")
    print(f"  Eval time:          {elapsed:.1f}s")
    print(f"{'-' * 60}")
    print(f"  Action MSE (simple avg):   {simple_avg_mse:.6f} ± {simple_std_mse:.6f}")
    print(f"  Action MSE (weighted avg): {weighted_mse:.6f}")
    print(f"  Action MSE (median):       {np.median(all_mse_arr):.6f}")
    print(f"  Action MSE (min):          {np.min(all_mse_arr):.6f}")
    print(f"  Action MSE (max):          {np.max(all_mse_arr):.6f}")
    print(f"{'=' * 60}")

    # Append summary rows to CSV
    if csv_file is not None:
        csv_writer.writerow([])
        csv_writer.writerow(
            ["summary_type", "", "", "action_mse", "action_mse_std"]
        )
        csv_writer.writerow(
            ["simple_avg", "", "", f"{simple_avg_mse:.6f}", f"{simple_std_mse:.6f}"]
        )
        csv_writer.writerow(
            ["weighted_avg", "", "", f"{weighted_mse:.6f}", ""]
        )
        csv_writer.writerow(
            ["median", "", "", f"{np.median(all_mse_arr):.6f}", ""]
        )
        csv_file.close()
        print(f"\n  Results saved to {args.save_csv_path}")


if __name__ == "__main__":
    main()
