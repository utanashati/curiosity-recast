import cv2
import gym
import vizdoomgym
import numpy as np
from gym.spaces.box import Box

import env_wrapper
import time

from collections import deque


# Modified from envs.py in https://github.com/pathak22/noreward-rl
def create_doom_env(env_id, rank, envWrap=True,
                    noLifeReward=False, acRepeat=0,
                    num_skip=4, num_stack=4):
    if 'very' in env_id.lower():
        env_id = 'VizdoomMyWayHomeVerySparse-v0'
    elif 'sparse' in env_id.lower():
        env_id = 'VizdoomMyWayHomeSparse-v0'
    else:
        env_id = 'VizdoomMyWayHomeDense-v0'

    # VizDoom workaround: Simultaneously launching multiple vizdoom processes
    # makes program stuck, so use the global lock in multi-threading/processing
    rank = int(rank)
    time.sleep(rank * 10)
    env = gym.make(env_id)
    env.reset()
    # acwrapper = wrappers.ToDiscrete('minimal')
    # env = acwrapper(env)
    # env = env_wrapper.MakeEnvDynamic(env)  # to add stochasticity

    if envWrap:
        if noLifeReward:
            env = env_wrapper.NoNegativeRewardEnv(env)
        env = PreprocessFrames(env, num_skip=num_skip)
        if num_stack > 1:
            env = StackFrames(env, num_stack=num_stack)
    elif noLifeReward:
        env = env_wrapper.NoNegativeRewardEnv(env)

    return env


# Taken from https://github.com/amld/workshop-Artificial-Curiosity/blob/master/src/environments.py
class PreprocessFrames(gym.Wrapper):
    """ Skip, normalize and resize original frames from the environment """

    def __init__(self, env, num_skip=4):  # size=84
        super(PreprocessFrames, self).__init__(env)
        self.num_skip = num_skip
        # self.size = size
        self.unwrapped.original_size = self.env.observation_space.shape
        self.env.observation_space.shape = (1, 42, 42)

    def preprocess_frame(self, frame):
        frame = np.mean(frame, axis=2)
        # frame = cv2.resize(frame, (self.size, self.size))
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (42, 42))
        frame = frame.astype(np.float32) / 255.0
        frame = frame.reshape(1, 42, 42)  # self.size, self.size)
        return frame

    def step(self, action):
        _observation, reward, done, info = self.env.step(action)
        actual_steps = 1
        while actual_steps < self.num_skip and not done:
            _observation, _reward, done, info = self.env.step(action)
            reward += _reward
            actual_steps += 1
        self.unwrapped.original_observation = _observation.copy()
        observation = self.preprocess_frame(_observation.copy())
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.unwrapped.original_observation = observation
        obs = self.preprocess_frame(observation)
        return obs


class StackFrames(gym.Wrapper):
    """ Stack consecutive frames together """

    def __init__(self, env, num_stack=4):
        super(StackFrames, self).__init__(env)
        self.num_stack = num_stack
        self.stack = deque(maxlen=num_stack)

    def step(self, action):
        _observation, reward, done, info = self.env.step(action)
        self.stack.append(_observation)
        observation = np.concatenate(self.stack, axis=0)
        return observation, reward, done, info

    def reset(self):
        _observation = self.env.reset()
        for i in range(self.stack.maxlen):
            self.stack.append(_observation)
        observation = np.concatenate(self.stack, axis=0)
        return observation


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
