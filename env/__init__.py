from gymnasium.envs.registration import register

register(
    id="CartPoleadv-v1",
    entry_point="env.cart_pole.ENV_V1:CartPoleEnv_adv",  # Adjust as per your directory structure
)

# Register the environment
register(
    id="CustomInvertedPendulum-v0",  # Unique ID
    entry_point="env.mujoco_inv_pend.cost_pend:CustomInvertedPendulumEnv",  # Path to your custom environment
)

register(
    id="HalfCheetahCost-v0",  # Unique ID
    entry_point="env.mujoco_half_cheetah.half_cheetah_cost:HalfCheetahEnv_lya",  # Path to your custom environment
)
