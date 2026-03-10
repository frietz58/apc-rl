import numpy as np
import gymnasium as gym
from .pointmaze_layouts import *  # bad practice but just to quickly get the different maze layouts 

class FrankaKitchenDenseRewardPretrainWrapper(gym.Wrapper):
    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.compute_dense_reward(obs, action)
        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs, action):
        finger_1_xpos = self.env.unwrapped.robot_env.data.body("panda0_leftfinger").xpos
        finger_2_xpos = self.env.unwrapped.robot_env.data.body("panda0_rightfinger").xpos
        ee_pos = (finger_1_xpos + finger_2_xpos) / 2.0

        if len(self.env.unwrapped.tasks_to_complete) >= 1:
            target_obj = next(iter(self.env.unwrapped.tasks_to_complete))  # hack to get item from set
            if target_obj == "hinge cabinet":
                target_obj = "hingecab"
            elif target_obj == "slide cabinet":
                target_obj = "slidecabinet"
            elif target_obj == "light switch":
                target_obj = "lightswitchroot"
            elif target_obj == "top burner":
                target_obj = "knob 3"
            elif target_obj == "bottom burner":
                target_obj = "knob 1"

            target_pos = self.env.unwrapped.robot_env.data.body(target_obj).xpos

            reward_ee_dist = - 0.5 * np.linalg.norm(ee_pos - target_pos)

            if target_obj == "hingecab":
                reward_ee_dist = - 0.1 * np.linalg.norm(ee_pos - target_pos)

            achieved_goal = obs["achieved_goal"]
            desired_goal = obs["desired_goal"]
            task_dist_reward = 0.0
            for task in self.env.unwrapped.tasks_to_complete:
                distance = np.linalg.norm(achieved_goal[task] - desired_goal[task])
                task_dist_reward -= distance

            todo_tasks_reward = 0

            reward = reward_ee_dist + task_dist_reward + todo_tasks_reward

        elif len(self.env.unwrapped.tasks_to_complete) == 0:
            reward = 100
        
        return reward


def make_env(
        env_id, 
        seed, 
        env_args,
        render_mode="rgb_array"
):
    def thunk():
        if env_id.startswith("PointMaze"):
            pointmaze_map_layout = env_args.get("env_task", "gtl")
            if pointmaze_map_layout == "gtl":
                maze_map = GTL
            elif pointmaze_map_layout == "gtr":
                maze_map = GTR
            elif pointmaze_map_layout == "gbr":
                maze_map = GBR
            elif pointmaze_map_layout == "gbl":
                maze_map = GBL
            else:
                raise ValueError(f"Unknown pointmaze map layout: {pointmaze_map_layout}")

            env = gym.make(
                env_id,
                maze_map=maze_map,
                continuing_task=False,
                reward_type="dense",
                max_episode_steps=env_args.get("env_ep_len", 400),
                render_mode=render_mode,
            )

            env = gym.wrappers.TransformReward(env, lambda r: r - 1)  # small constant punishment to make it finish quickly, used for main-body results
        
        elif env_id == "FrankaKitchen-v1":
            env = gym.make(
                'FrankaKitchen-v1', 
                tasks_to_complete=[env_args.get("env_task", "microwave")],
                render_mode=render_mode
                )

            env = FrankaKitchenDenseRewardPretrainWrapper(env)

        else:
            env = gym.make(env_id, render_mode=render_mode)
                
        env = gym.wrappers.RecordEpisodeStatistics(env)

        obs_transform = env_args.get("env_obs_transform", "flatten")
        if obs_transform == "flatten":
            env = gym.wrappers.FlattenObservation(env)
        elif obs_transform == "obs":
            env = gym.wrappers.TransformObservation(env, lambda obs: obs['observation'], observation_space=env.observation_space["observation"])
        elif obs_transform == "None":
            assert env_id.startswith("MyCarRacing"), "obs_transform='None' should be passed when using MyCarRacing env."
            pass
        else:
            raise NotImplementedError

        env.action_space.seed(seed)
        return env

    return thunk