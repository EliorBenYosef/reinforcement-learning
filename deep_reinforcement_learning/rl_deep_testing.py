from deep_reinforcement_learning.rl_deep import LIBRARY_TF, LIBRARY_TORCH, LIBRARY_KERAS
from deep_reinforcement_learning.rl_deep import DQL, PG, AC, DDPG


if __name__ == '__main__':
    lib_type = LIBRARY_TF         # LIBRARY_TF, LIBRARY_TORCH, LIBRARY_KERAS
    DQL.play(0, lib_type)         # 0-CartPole, 1-Breakout, 2-SpaceInvaders
    # PG.play(0, lib_type)          # 0-CartPole, 1-Breakout, 2-SpaceInvaders
    # AC.play(0, lib_type)          # 0-CartPole, 1-Pendulum, 2-MountainCarContinuous
    # DDPG.play(0, lib_type)        # 0-Pendulum, 1-MountainCarContinuous
    pass
