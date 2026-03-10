
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
from nflow.utils import make_flow, IdentityFlow
from sac.sac_agent import SACAgentContinuous


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


def load_pretrained_flows(envs, flow_paths, device, include_identity_flow=False):
    pretrained_flows = []
    
    if include_identity_flow:
        pretrained_flows.append(IdentityFlow())
    
    for flow_path in flow_paths:
        flow_config = yaml.load(open(f"{flow_path}/config.yaml"), Loader=yaml.FullLoader)
        flow = make_flow(
            action_dim=envs.single_action_space.shape[0],
            state_dim=envs.single_observation_space.shape[0],
            num_layers=flow_config["flow_num_layers"],
            hidden_dim=flow_config["flow_hidden_dim"],
            device=device,
        )
        flow.load_state_dict(torch.load(f"{flow_path}/flow.pt"))
        pretrained_flows.append(flow)
    return pretrained_flows


def select_random_action(envs, flows, obs, device):
    selector_actions = np.random.randint(0, len(flows), size=(envs.num_envs,))

    all_flow_actions = []
    all_latent_actions = []
    for f_i in range(len(flows)):
        if flows[f_i].base_dist is None:
            # identity dummy has no base dist to sample from....
            latent_actions = torch.from_numpy(np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])).float().to(device)  # (n_envs, actions)
        else:
            latent_actions = flows[f_i].base_dist.sample((envs.num_envs,))  # (n_envs, actions)
            
        flow_actions, _ = flows[f_i].latent_to_real(latent_actions.float().to(device), context=torch.from_numpy(obs).float())
        all_latent_actions.append(latent_actions)
        all_flow_actions.append(flow_actions)

    all_flow_actions = torch.stack(all_flow_actions)  # (n_flows, n_envs, actions)
    env_actions = all_flow_actions[selector_actions, torch.arange(envs.num_envs)]  # (n_envs, actions)

    # make the latent action corresponding to the env actions by using the inverse transformation!
    env_latent_actions = []
    for f_i in range(len(flows)):
        inverse_latent_actions, _ = flows[f_i].real_to_latent(env_actions, context=torch.from_numpy(obs).float())
        env_latent_actions.append(inverse_latent_actions)

    return env_actions, env_latent_actions


def select_apc_action(flows, per_flow_agents, obs, device, selector_temp):
    all_flow_actions = []
    all_latent_actions = []
    all_flow_qvals = []
    for f_i in range(len(flows)):
        latent_actions = per_flow_agents[f_i].get_action(torch.Tensor(obs).to(device))

        q1_t = per_flow_agents[f_i].qf1(torch.Tensor(obs).to(device), latent_actions)
        q2_t = per_flow_agents[f_i].qf2(torch.Tensor(obs).to(device), latent_actions)
        min_q_t = torch.min(q1_t, q2_t)
        all_flow_qvals.append(min_q_t)

        flow_actions, _ = flows[f_i].latent_to_real(latent_actions, context=torch.from_numpy(obs).float())
        all_latent_actions.append(latent_actions)
        all_flow_actions.append(flow_actions)
        
    all_flow_actions = torch.stack(all_flow_actions)  # (n_flows, n_envs, actions)
    all_qs = torch.stack(all_flow_qvals)  # (n_flows, n_envs, 1)

    # use softmax over Q values to select the action
    selector_dist = torch.distributions.Categorical(logits=(all_qs / selector_temp).squeeze(-1).T)
    selector_actions = selector_dist.sample()

    env_actions = all_flow_actions[selector_actions, torch.arange(envs.num_envs)]
    env_latent_actions = []
    for f_i in range(len(flows)):
        inverse_latent_actions, _ = flows[f_i].real_to_latent(env_actions, context=torch.from_numpy(obs).float())
        env_latent_actions.append(inverse_latent_actions)

    selector_actions = selector_actions.detach().cpu().numpy()
    
    return env_actions, env_latent_actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/car_racing.yaml", help="Default config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--parrot", action="store_true", help="If true use PARROT baseline instead of APC", default=False)
    parser.add_argument("--pretrained_flow_paths", type=str, nargs="+", default=None, help="List of paths to pretrained normalizing flow behavior prior(s)")
    parser.add_argument("--env_task", type=str, default=None, help="Environment task to use (overrides config file)")
    parser.add_argument("--tag", type=str, default="", help="Optional tag to add to the save directory name")
    args = parser.parse_args()
    
    # load config
    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
        
        # override default args 
        raw_config["seed"] = args.seed
        if args.pretrained_flow_paths is not None:
            raw_config["apc_flow_paths"] = args.pretrained_flow_paths
        if args.parrot:
            raw_config["apc_use_parrot"] = True
        if args.env_task is not None:
            raw_config["env_task"] = args.env_task
            
    config = dict_to_namespace(raw_config)
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if config.apc_use_parrot:
        variant_name = "OnlinePARROT"
        assert len(config.apc_flow_paths) == 1, "PARROT baseline only supports a single pretrained flow as behavior prior"
        
    else:
        variant_name = "OnlineAPC"
    
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
    
    pretrained_flows = []    
    if config.apc_use_parrot:
        # PARROT baseline only support a single pretrained flow behavior prior
        pretrained_flows = load_pretrained_flows(envs, [config.apc_flow_paths[0]], device)
    else:
        # APC loads all given pretrained flows and include identity flow for prior-free actor
        pretrained_flows = load_pretrained_flows(envs, config.apc_flow_paths, device, include_identity_flow=True)        
    
    # create a SAC agent for each pretrained flow
    per_flow_agents = []
    for i in range(len(pretrained_flows)):
        agent_i = SACAgentContinuous(
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
            il_dataset_sample_fun=None,
        )
        per_flow_agents.append(agent_i)
        
    obs, _ = envs.reset(seed=config.seed)
    successes, returns, lengths, steps = [], [], [], []

    for global_step in range(config.sac_total_timesteps): 
        if global_step < config.sac_learning_starts:
            # random explore
            real_actions, latent_actions = select_random_action(envs, pretrained_flows, obs, device)
        else:
            # apc select actions
            real_actions, latent_actions = select_apc_action(pretrained_flows, per_flow_agents, obs, device, config.apc_selector_boltzmann_temp)
        
        # step env
        real_actions = real_actions.detach().cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = envs.step(real_actions)
        
        # handle episode end
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

        # add LATENT actions to per-agent replay buffers
        for a_idx in range(len(per_flow_agents)):
            per_flow_agents[a_idx].buffer_add(obs, real_next_obs, latent_actions[a_idx].detach().cpu().numpy(), rewards, terminations, None)
            
        obs = next_obs
        
        # train step
        if global_step >= config.sac_learning_starts:
            for agent_idx in range(len(per_flow_agents)):
                qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss, actor_loss, alpha, alpha_loss = per_flow_agents[agent_idx].train()

                if global_step % 100 == 0:
                    writer.add_scalar(f"losses/agent{agent_idx}/qf1_values", qf1_a_values.mean().item(), global_step * config.sac_parallel_envs )
                    writer.add_scalar(f"losses/agent{agent_idx}/qf2_values", qf2_a_values.mean().item(), global_step * config.sac_parallel_envs )
                    writer.add_scalar(f"losses/agent{agent_idx}/qf1_loss", qf1_loss.item(), global_step * config.sac_parallel_envs)
                    writer.add_scalar(f"losses/agent{agent_idx}/qf2_loss", qf2_loss.item(), global_step * config.sac_parallel_envs)
                    writer.add_scalar(f"losses/agent{agent_idx}/qf_loss", qf_loss.item() / 2.0, global_step * config.sac_parallel_envs)
                    writer.add_scalar(f"losses/agent{agent_idx}/actor_loss", actor_loss.item(), global_step * config.sac_parallel_envs)
                    writer.add_scalar(f"losses/agent{agent_idx}/alpha", alpha, global_step * config.sac_parallel_envs)
                    if config.sac_autotune_alpha:
                        writer.add_scalar(f"losses/agent{agent_idx}/alpha_loss", alpha_loss.item(), global_step * config.sac_parallel_envs)

        # periodically save the model
        if global_step % 10000 == 0:
            for agent_idx in range(len(per_flow_agents)):
                per_flow_agents[agent_idx].save_models(f"{save_dir}/sac_agent{agent_idx}")
 
    envs.close()
    writer.close()
    
    for agent_idx in range(len(per_flow_agents)):
        per_flow_agents[agent_idx].save_models(f"{save_dir}/sac_agent{agent_idx}")
        
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
    
    print(f"Training completed at time {time.strftime('%Y-%m-%d %H:%M:%S')}. Models and logs saved to {save_dir}")
                
        