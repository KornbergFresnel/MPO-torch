from .wrappers import EnvWrapper


def create_env_wrapper(config):
    env_name = config["env"]
    # if env_name == "Pendulum-v0":
    #     return PendulumWrapper(config)
    # elif env_name == "BipedalWalker-v2":
    #     return BipedalWalker(config)
    # elif env_name == "LunarLanderContinuous-v2":
    #     return LunarLanderContinous(config)
    return EnvWrapper(env_name)
