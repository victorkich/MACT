from configs.Config import Config
from env.starcraft.StarCraft import StarCraft


class EnvConfig(Config):
    def __init__(self):
        pass

    def create_env(self):
        pass


class StarCraftConfig(EnvConfig):

    def __init__(self, env_name, seed):
        self.env_name = env_name
        self.seed = seed

    def create_env(self):
        return StarCraft(self.env_name, self.seed)
