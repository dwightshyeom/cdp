'''
python robomimic/scripts/cdp_run_trained_agent.py \
            --ckpt /home/arclab/workspace/robomimic/cdp_trained_models/cdp_box_push_cfg/20260314120154/models/model_epoch_500.pth \
            --target_dist 0.10 \
            --cond_lambda 2.0 \
            --n_rollouts 20 \
            --video_path ./videos/10_200_200_500/rollout_10_200_200_500_20.mp4

'''

import argparse
import json
import os
import numpy as np
import torch
import imageio

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils

def inject_push_distance(obs, env):
    obs["push_distance"] = np.array(
        [env.env.target_push_distance], dtype=np.float32
    )
    return obs


def run_rollout(policy, env, target_dist, cond_lambda, horizon, video_path, video_skip, camera_name):
    """Run a single rollout and save a video."""

    with policy.policy.global_config.values_unlocked():
        policy.policy.algo_config.cfg.cond_lambda = cond_lambda

    # Set target distance in the environment
    if hasattr(env.env, 'set_target_distance'):
        env.env.set_target_distance(target_dist)
    else:
        print("[!] Warning: could not find set_target_distance on the environment.")

    print(f"\n>>> Running Rollout | Target: {target_dist}m | Lambda: {cond_lambda} <<<")

    policy.start_episode()
    obs = env.reset()
    obs = inject_push_distance(obs, env)

    video_writer = None
    if video_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)
        print(f"[+] Recording video to: {video_path}")

    step_count = 0
    success = False
    video_count = 0
    success_hold_required = 200
    success_hold_count = success_hold_required

    try:
        while step_count < horizon:
            if video_writer is not None and video_count % video_skip == 0:
                frame = env.render(mode="rgb_array", height=512, width=512, camera_name=camera_name)
                video_writer.append_data(frame)
            video_count += 1

            # Query policy and step simulation
            action = policy(ob=obs)
            obs, reward, done, _ = env.step(action)
            obs = inject_push_distance(obs, env)
            step_count += 1

            if step_count % 50 == 0:
                print(f"  Step {step_count}/{horizon} ...")

            # Success state machine: box must hold position for success_hold_required consecutive steps
            if hasattr(env.env, '_check_success') and env.env._check_success():
                success_hold_count -= 1
                print(f"  [~] Success condition met! Hold for {success_hold_count} more steps...")
                if success_hold_count <= 0:
                    print(f"  [✓] Success at step {step_count}! Box reached and held {target_dist}m.")
                    success = True
                    break
            else:
                success_hold_count = success_hold_required  # reset if box leaves target zone

            if done:
                break

    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"[+] Video saved to: {os.path.abspath(video_path)}")

    if not success:
        print(f"  [-] Rollout ended at step {step_count} without reaching target distance.")

    return success, step_count


def run_evaluation(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[+] Loading model from {args.ckpt} onto {device}...")

    # Load policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.ckpt,
        device=device,
    )

    # Create environment
    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )

    if args.target_dist is not None:
        all_success = []
        all_steps = []

        for rollout_idx in range(args.n_rollouts):
            if args.video_path is not None:
                if args.n_rollouts == 1:
                    # Single rollout
                    video_path = args.video_path
                else:
                    # Multiple rollouts
                    base, ext = os.path.splitext(args.video_path)
                    video_path = f"{base}_{rollout_idx:03d}{ext}"
            else:
                video_path = None

            print(f"\n[Rollout {rollout_idx + 1}/{args.n_rollouts}]")
            success, steps = run_rollout(
                policy=policy,
                env=env,
                target_dist=args.target_dist,
                cond_lambda=args.cond_lambda,
                horizon=args.horizon,
                video_path=video_path,
                video_skip=args.video_skip,
                camera_name=args.camera_name,
            )
            all_success.append(float(success))
            all_steps.append(steps)

        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        summary = {
            "Num_Rollouts": args.n_rollouts,
            "Target_Distance": args.target_dist,
            "CFG_Lambda": args.cond_lambda,
            "Num_Success": int(sum(all_success)),
            "Success_Rate": float(np.mean(all_success)),
            "Avg_Steps": float(np.mean(all_steps)),
        }
        print(json.dumps(summary, indent=4))

        # Save summary to JSON
        if args.video_path is not None:
            abs_video_path = os.path.abspath(args.video_path)
            video_dir = os.path.dirname(abs_video_path)
            parent_dir = os.path.dirname(video_dir)

            file_name = os.path.splitext(os.path.basename(abs_video_path))[0]
            json_path = os.path.join(parent_dir, f"{file_name}.json")

            os.makedirs(parent_dir, exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"\n[+] Saved evaluation summary to: {json_path}")
    else:
        # Interactive mode
        try:
            rollout_idx = 0
            while True:
                dist_str = input("\n[?] Enter target push distance (e.g. 0.1, 0.15) or 'q' to quit: ")
                if dist_str.lower() == 'q':
                    break
                try:
                    target_dist = float(dist_str)
                except ValueError:
                    print("[!] Invalid input.")
                    continue

                lambda_str = input("[?] Enter CFG lambda (ENTER for default 2.0): ")
                try:
                    cond_lambda = float(lambda_str) if lambda_str.strip() != "" else 2.0
                except ValueError:
                    cond_lambda = 2.0

                video_path = None
                os.mkdir(args.video_dir, exist_ok=True)
                if args.video_dir is not None:
                    video_path = os.path.join(
                        args.video_dir,
                        f"rollout_{rollout_idx:03d}_dist{target_dist}_lam{cond_lambda}.mp4"
                    )

                run_rollout(
                    policy=policy,
                    env=env,
                    target_dist=target_dist,
                    cond_lambda=cond_lambda,
                    horizon=args.horizon,
                    video_path=video_path,
                    video_skip=args.video_skip,
                    camera_name=args.camera_name,
                )
                rollout_idx += 1

        except KeyboardInterrupt:
            print("\n[!] Interrupted.")

    env.env.close()
    print("Environment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained CDP checkpoint (.pth)",
    )
    parser.add_argument(
        "--target_dist",
        type=float,
        default=None,
        help="(non-interactive) target push distance in meters",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=1,
        help="Number of rollouts to run in batch mode (default: 1)",
    )
    parser.add_argument(
        "--cond_lambda",
        type=float,
        default=2.0,
        help="CFG guidance scale (default: 2.0)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
        help="Max rollout steps (default: 1000)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(non-interactive) save rollout video to this path",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="(interactive) directory to save per-rollout videos",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="Record one video frame every N steps (default: 5)",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="agentview",
        help="Camera to use for video recording (default: agentview)",
    )
    args = parser.parse_args()
    run_evaluation(args)