""" RL env runner """
from collections import defaultdict

import numpy as np


class EnvRunner:
    """ Reinforcement learning runner in an environment with given policy """

    def __init__(self, env, policy, nsteps, transforms=None, step_var=None):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or []
        self.step_var = step_var if step_var is not None else 0
        self.latest_observation = self.env.reset()

    @property
    def nenvs(self):
        """ Returns number of batched envs or `None` if env is not batched """
        return getattr(self.env.unwrapped, "nenvs", None)
    
    def write(self, name, val):
        """ Writes logs """
        if type(val) is dict:
            self.env.writer.add_scalars(name, val, self.env.step_var)
        else:
            self.env.writer.add_scalar(name, val, self.env.step_var)

    def reset(self):
        """ Resets env and runner states. """
        self.latest_observation = self.env.reset()
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        dones = []
        
        for i in range(self.nsteps):
            observations.append(self.latest_observation)
            act = self.policy.act(self.latest_observation)
            
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            assert type(act["actions"]) is np.ndarray, "Error: actions for environment must be numpy"
                
            for key, val in act.items():
                trajectory[key].append(val)

            obs, rew, done, _ = self.env.step(trajectory["actions"][-1])
            self.latest_observation = obs
            rewards.append(rew)
            dones.append(done)
            self.step_var += self.nenvs or 1

            # Only reset if the env is not batched. Batched envs should
            # auto-reset.
            if not self.nenvs and np.all(done):
                self.latest_observation = self.env.reset()

        trajectory.update(
            observations=observations,
            rewards=rewards,
            dones=dones)

        for transform in self.transforms:
            transform(trajectory, self.latest_observation)
        
        return trajectory
