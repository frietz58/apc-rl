# based on clean_rl
import argparse
import os
import random
import time
import yaml
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
# import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import envs  # to register the custom environments
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from envs.utils import make_env
from sac.sac_agent import SACAgentContinuous
from sac.utils import make_demonstration_dataset


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/maze.yaml", help="Default config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--il_coef", type=float, default=None, help="Imitation learning coefficient (overrides config file)")
    parser.add_argument("--il_use_q_filter", action="store_true", help="Whether to use Q-filter for imitation learning")
    parser.add_argument("--il_data_paths", type=str, nargs="+", default=None, help="List of paths to imitation learning datasets (overrides config file)")
    parser.add_argument("--env_task", type=str, default=None, help="Environment task to use (overrides config file)")
    parser.add_argument("--tag", type=str, default="", help="Optional tag to add to the save directory name")
    args = parser.parse_args()
    
    # load config
    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
        
        # override default args 
        raw_config["seed"] = args.seed
        if args.il_coef is not None:
            raw_config["sac_il_coef"] = args.il_coef
        if args.il_data_paths is not None:
            raw_config["sac_il_data_paths"] = args.il_data_paths
        if args.il_use_q_filter:
            raw_config["sac_il_q_filter"] = True
        if args.env_task is not None:
            raw_config["env_task"] = args.env_task
            
    config = dict_to_namespace(raw_config)
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    variant_name = "OnlineSAC"
    if config.sac_il_coef > 0.0:
        variant_name += f"-IL{config.sac_il_coef}"
    if config.sac_il_q_filter:
        variant_name += "-WithQFilter"
        
    if config.env_id.startswith("PointMaze"):
        env_str = "Maze"
    elif config.env_id.startswith("FrankaKitchen"):
        env_str = "Kitchen"
    elif config.env_id.startswith("MyCarRacing"):
        env_str = "CarRacing"

    save_dir = f"trained_agents/{env_str}/{variant_name}-{ts}"
    
    if args.tag:
        save_dir += f"-{args.tag}"
        
    os.makedirs(save_dir, exist_ok=True)

    # save the config
    with open(f"{save_dir}/config.yaml", "w") as f:
        yaml.dump(vars(config), f)
    
    writer = SummaryWriter(f"{save_dir}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )
    
    # env setup
    env_config = {}
    for k, v in vars(config).items():
        if k.startswith("env"):
            env_config[k] = v
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(
            config.env_id,
            config.seed + i,
            render_mode="rgb_array",
            env_args=env_config
            
        ) for i in range(config.sac_parallel_envs)],
        autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    if config.sac_il_coef > 0.0:
        il_dataset_sample_fun = make_demonstration_dataset(config.sac_il_data_paths)
    else:
        il_dataset_sample_fun = None

    # agent setup
    sac_agent = SACAgentContinuous(
        envs=envs,
        device=device,
        alpha=config.sac_entropy_alpha,
        autotune_alpha=config.sac_autotune_alpha,
        q_lr=config.sac_critic_lr,
        policy_lr=config.sac_actor_lr,
        buffer_size=config.sac_buffer_size,
        batch_size=config.sac_batch_size,
        gamma=config.sac_discount_factor,
        target_tau=config.sac_polyak_factor,
        il_coef=config.sac_il_coef,
        il_dataset_sample_fun=il_dataset_sample_fun,
        il_use_q_filter=config.sac_il_q_filter,
    )
    

    obs, _ = envs.reset(seed=config.seed)
    successes, returns, lengths, steps = [], [], [], []
    for global_step in range(config.sac_total_timesteps):
        if global_step < config.sac_learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = sac_agent.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # handle episode terminattion and truncations
        real_next_obs = next_obs.copy()
        for idx in range(config.sac_parallel_envs):
            if truncations[idx] or terminations[idx]:
                real_next_obs[idx] = infos["final_obs"][idx]

                writer.add_scalar("charts/episodic_return", infos["final_info"]["episode"]["r"][idx], global_step * config.sac_parallel_envs + idx)
                writer.add_scalar("charts/episodic_length", infos["final_info"]["episode"]["l"][idx], global_step * config.sac_parallel_envs + idx)
                if "success" in infos["final_info"]:
                    if infos["final_info"]["success"][idx] is None:
                        success = False
                    else:
                        success = bool(infos["final_info"]["success"][idx])
                else:
                    success = terminations[idx]
                successes.append(success)
                writer.add_scalar("charts/ep_success", success, global_step * config.sac_parallel_envs + idx)
                if len(successes) > 100:
                    writer.add_scalar("charts/success_rate", sum(successes[-100:]) / 100, global_step * config.sac_parallel_envs + idx)

                print(f"global_step={global_step}, episodic_return={infos["final_info"]["episode"]["r"][idx]:.2f}, episode_len={infos["final_info"]["episode"]["l"][idx]}, success={success}")

                returns.append(infos["final_info"]["episode"]["r"][idx])
                lengths.append(infos["final_info"]["episode"]["l"][idx])
                steps.append(global_step * config.sac_parallel_envs + idx)
                
        sac_agent.buffer_add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # training
        if global_step > config.sac_learning_starts:
            qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss, actor_loss, alpha, alpha_loss = sac_agent.train()

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step * config.sac_parallel_envs)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step * config.sac_parallel_envs)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step * config.sac_parallel_envs)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step * config.sac_parallel_envs)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step * config.sac_parallel_envs)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step * config.sac_parallel_envs)
                writer.add_scalar("losses/alpha", alpha, global_step * config.sac_parallel_envs)
                if config.sac_autotune_alpha:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step * config.sac_parallel_envs)
                    
            # periodically save agent
            if global_step % 10000 == 0:
                sac_agent.save_models(f"{save_dir}/checkpoints")
        
    envs.close()
    writer.close()
    
    sac_agent.save_models(f"{save_dir}/checkpoints")
    
    # make simple plot 
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.subplot(1, 3, 2)
    plt.plot(lengths)
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.subplot(1, 3, 3)
    plt.plot(successes)
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_plots.png")
    plt.close()

    # save logs to disk
    np.savez(f"{save_dir}/training_logs.npz", returns=returns, lengths=lengths, successes=successes, steps=steps)
    
    print(f"Training completed. Models and logs saved to {save_dir}")
    