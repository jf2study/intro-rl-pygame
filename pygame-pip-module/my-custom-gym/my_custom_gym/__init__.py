from gym.envs.registration import register
#

register(
    id='custom-py-game-env-v0',
    entry_point='my_custom_gym.envs:CustomPyGameEnv',
)

