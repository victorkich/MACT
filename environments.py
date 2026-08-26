from enum import Enum


class Env(str, Enum):
    """Environment identifiers.

    This release ships the StarCraft II code used for the paper. The other members are kept
    because the runner and worker branch on them, and removing them would mean rewriting
    control flow that is not exercised here.
    """
    STARCRAFT = "starcraft"
    FLATLAND = "flatland"
    PETTINGZOO = "pettingzoo"
    GRF = "football"
    MAMUJOCO = "mamujoco"


RANDOM_SEED = 23
