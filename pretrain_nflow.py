import argparse
import os
import yaml
import pickle
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from nflow.utils import make_flow


def load_data(flow_data_path, flow_batch_size, device):
    print("Loading expert data...")
    data = []
    # for path in flow_data_path:
    with open(flow_data_path, 'rb') as f:
        task_data = pickle.load(f)
        data.extend(task_data)

    print("Done loading data. Create dataset and data loader...")
    print(f"Number of episodes loaded: {len(data)}")

    state_dim = data[0]["states"].shape[1]
    action_dim = data[0]["actions"].shape[1]
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    # collect state action pairs
    all_states = []
    all_actions = []
    for traj in data:
        for s, a in zip(traj["states"], traj["actions"]):
            all_states.append(s)
            all_actions.append(a)

    all_states = np.stack(all_states)  # shape [N, state_dim]
    all_actions = np.stack(all_actions)  # shape [N, action_dim]
    print(f"Total number of state-action pairs: {all_states.shape[0]}")

    states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)

    dataset = TensorDataset(states_tensor, actions_tensor)
    loader = DataLoader(dataset, batch_size=flow_batch_size, shuffle=True)
    print("Done creating dataset and data loader. Creating flow...")
    
    return loader, state_dim, action_dim


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/car_racing.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--override_data_path", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if args.override_data_path:
        config["flow_data_path"] = args.override_data_path
    
    if args.seed is not None:
        config["seed"] = args.seed

    # make deterministic
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = config["torch_deterministic"]
    np.random.seed(config["seed"])
    
    ts = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f"trained_flows/{config['env_id']}-{ts}"
    if args.tag:
        save_dir += f"-{args.tag}/"
    else:
        save_dir += "/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader, state_dim, action_dim = load_data(config["flow_data_path"], config["flow_batch_size"], device)
    
    # create flow
    flow = make_flow(
        action_dim=action_dim,
        state_dim=state_dim,
        num_layers=config["flow_num_layers"],
        hidden_dim=config["flow_hidden_dim"],
        base_dist_scale=config["flow_base_dist_scale"],
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(flow.parameters(), lr=config["flow_lr"], weight_decay=config["flow_wd"])

    # train flow using MLE
    print("Done creating flow. Training...")
    all_losses = []
    for epoch in range(config["flow_training_epochs"]):
        ep_losses = []
        print(f"Epoch {epoch}...")
        for batch_states, batch_actions in data_loader:

            nll = -flow.log_prob(batch_actions, context=batch_states).mean()

            # regularize inverse-consistency
            eps_a = torch.randn_like(batch_actions) * config["flow_sigma_a"]
            with torch.no_grad():
                z_ref, _ = flow.real_to_latent(batch_actions, context=batch_states)
            z_noisy, _ = flow.real_to_latent(batch_actions + eps_a, context=batch_states)
            ic_pen = ((z_noisy - z_ref).pow(2).sum(dim=1) / (eps_a.pow(2).sum(dim=1) + 1e-8)).mean()

            # regularize forward-smoothness
            with torch.no_grad():
                z0, _ = flow.real_to_latent(batch_actions, context=batch_states)
            delta_z = torch.randn_like(z0) * config["flow_sigma_z"]
            a0, _ = flow.latent_to_real(z0, context=batch_states)
            a1, _ = flow.latent_to_real(z0 + delta_z, context=batch_states)
            fs_pen = ((a1 - a0).pow(2).sum(dim=1) / (delta_z.pow(2).sum(dim=1) + 1e-8)).mean()

            loss = nll + config["flow_lambda_ic"] * ic_pen + config["flow_lambda_fs"] * fs_pen

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=config["flow_grad_clip"])
            optimizer.step()
            ep_losses.append(loss.item())
            
        all_losses.append(np.mean(ep_losses))
        print(f"Epoch {epoch + 1}/{config['flow_training_epochs']}, Loss: {np.mean(ep_losses):.4f}")
        
    print("Done training flow.")
    
    # save to disk
    flow_save_path = os.path.join(save_dir, "flow.pt")
    torch.save(flow.state_dict(), flow_save_path)
    
    # plot loss curve
    import matplotlib.pyplot as plt
    plt.plot(all_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Flow Training Loss Curve")
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()
    
    print("Flow saved to:")
    print(flow_save_path)
    
