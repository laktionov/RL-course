# taken from OpenAI baselines.

import numpy as np
from gymnasium import Wrapper, RewardWrapper, ObservationWrapper
from gymnasium.spaces import Box


class MaxAndSkipEnv(Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FireResetEnv(Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, _ = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, {}


# in torch imgs have shape [c, h, w] instead of common [h, w, c]
class AntiTorchWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.img_size = [env.observation_space.shape[i]
                         for i in [1, 2, 0]
                         ]
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def observation(self, img):
        """what happens to each observation"""
        img = img.transpose(1, 2, 0)
        return img
