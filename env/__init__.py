from gymnasium.envs.registration import register

register(
    id="CartPoleadv-v1",
    entry_point="env.cart_pole.ENV_V1:CartPoleEnv_adv",  # Adjust as per your directory structure
)
