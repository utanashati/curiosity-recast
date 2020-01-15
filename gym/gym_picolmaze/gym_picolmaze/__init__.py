from gym.envs.registration import register

register(
    id='picolmaze-v0',
    entry_point='gym_picolmaze.envs:PicolmazeEnv',
)
