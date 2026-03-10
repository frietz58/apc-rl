from gymnasium import register

register(
    id=f"MyCarRacing-v1",
    entry_point="envs.car_racing_gym:CarRacing",
    kwargs={
        "obs_mode": "symbolic"
    }
)
