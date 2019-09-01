from deep_reinforcement_learning.rl_deep import LIBRARY_TF, LIBRARY_TORCH, LIBRARY_KERAS
from deep_reinforcement_learning.rl_deep import DQL, PG, AC, DDPG


if __name__ == '__main__':
    lib_type = LIBRARY_TF         # LIBRARY_TF, LIBRARY_TORCH, LIBRARY_KERAS
    DQL.play(0, lib_type)         # CartPole (0), Breakout (1), SpaceInvaders (2)
    PG.play(0, lib_type)          # CartPole (0), Breakout (1), SpaceInvaders (2)
    AC.play(0, lib_type)          # CartPole (0), Pendulum (1), MountainCarContinuous (2)
    DDPG.play(0, lib_type)        # Pendulum (0), MountainCarContinuous (1)
    pass
