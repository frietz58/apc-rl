import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset


def make_demonstration_dataset(data_paths):
    # load the expert data
    print("Loading IL data...")
    data = []
    for path in data_paths:
        with open(path, 'rb') as f:
            task_data = pickle.load(f)
            data.extend(task_data)

    # collect state action pairs
    all_states = []
    all_actions = []
    for traj in data:
        for s, a in zip(traj["states"], traj["actions"]):
            all_states.append(s)
            all_actions.append(a)

    all_states = np.stack(all_states)  # shape [N, state_dim]
    all_actions = np.stack(all_actions)  # shape [N, action_dim]

    states_tensor = torch.tensor(all_states, dtype=torch.float32)
    actions_tensor = torch.tensor(all_actions, dtype=torch.float32)

    dataset = TensorDataset(states_tensor, actions_tensor)
    print("Done creating IL dataset")

    # Function to sample a random batch
    def sample_batch_fun(batch_size):
        indices = torch.randint(0, len(dataset), (batch_size,))
        batch = tuple(t[indices] for t in dataset.tensors)
        return batch

    return sample_batch_fun