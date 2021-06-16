from gym.envs.registration import register
#

register(
    id='my-pygame-env-v0',
    entry_point='my_custom_gym.envs:MyPyGameEnv',
)

