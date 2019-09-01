from utils import Utils
from tabular_methods.envs_dss import Envs_DSS
from tabular_methods.rl_tabular import TabularMethods


class AlgorithmsTesting:

    @staticmethod
    def test_mc_policy_evaluation(episodes, print_v_table):
        # Mountain Car:
        car_vel_bin_num = 34  # ~100% success rate (32 rarely loses)
        mountain_car_env = Envs_DSS.MountainCar(Envs_DSS.MountainCar.CAR_VEL, car_vel_bin_num=car_vel_bin_num)
        mountain_car_model = TabularMethods.MonteCarloModel(mountain_car_env, episodes=episodes)
        # possible actions: backward (0), none (1), forward (2)
        mountain_car_policy = lambda velocity_state: 0 if velocity_state < (car_vel_bin_num//2) else 2
        mountain_car_model.perform_MC_policy_evaluation(mountain_car_policy, print_info=print_v_table)

        # Cart Pole:
        pole_theta_bin_num = 10
        cart_pole_env = Envs_DSS.CartPole(Envs_DSS.CartPole.POLE_THETA, pole_theta_bin_num=pole_theta_bin_num)
        cart_pole_model = TabularMethods.MonteCarloModel(cart_pole_env, episodes=episodes)
        # possible actions: left (0), right (1)
        cart_pole_policy = lambda theta_state: 0 if theta_state < (pole_theta_bin_num // 2) else 1
        cart_pole_model.perform_MC_policy_evaluation(cart_pole_policy, print_info=print_v_table)

        # Frozen Lake:
        frozen_lake_env = Envs_DSS.FrozenLake()
        frozen_lake_model = TabularMethods.MonteCarloModel(frozen_lake_env, episodes=episodes)
        frozen_lake_policy = lambda s: frozen_lake_env.env.action_space.sample()  # random policy
        frozen_lake_model.perform_MC_policy_evaluation(frozen_lake_policy, print_info=print_v_table)

    @staticmethod
    def test_td0_policy_evaluation(episodes, print_v_table):
        # Mountain Car:
        car_vel_bin_num = 34  # ~100% success rate (32 rarely loses)
        mountain_car_env = Envs_DSS.MountainCar(Envs_DSS.MountainCar.CAR_VEL, car_vel_bin_num=car_vel_bin_num)
        mountain_car_model = TabularMethods.TdZeroModel(mountain_car_env, episodes=episodes)
        # possible actions: backward (0), none (1), forward (2)
        mountain_car_policy = lambda velocity_state: 0 if velocity_state < (car_vel_bin_num//2) else 2
        mountain_car_model.perform_td0_policy_evaluation(mountain_car_policy, print_info=print_v_table)

        # Cart Pole:
        pole_theta_bin_num = 10
        cart_pole_env = Envs_DSS.CartPole(Envs_DSS.CartPole.POLE_THETA, pole_theta_bin_num=pole_theta_bin_num)
        cart_pole_model = TabularMethods.TdZeroModel(cart_pole_env, episodes=episodes)
        # possible actions: left (0), right (1)
        cart_pole_policy = lambda theta_state: 0 if theta_state < (pole_theta_bin_num//2) else 1
        cart_pole_model.perform_td0_policy_evaluation(cart_pole_policy, print_info=print_v_table)

        # Frozen Lake:
        frozen_lake_env = Envs_DSS.FrozenLake()
        frozen_lake_model = TabularMethods.TdZeroModel(frozen_lake_env, episodes=episodes)
        frozen_lake_policy = lambda s: frozen_lake_env.env.action_space.sample()  # random policy
        frozen_lake_model.perform_td0_policy_evaluation(frozen_lake_policy, print_info=print_v_table)

    @staticmethod
    def test_mc_non_exploring_starts_control(episodes, print_q_table_and_policy):
        method_name = 'MC non-exploring starts'

        # Frozen Lake:
        frozen_lake_env = Envs_DSS.FrozenLake()
        frozen_lake_model = TabularMethods.MonteCarloModel(frozen_lake_env, episodes=episodes,
                                                           eps_max=frozen_lake_env.EPS_MIN)
        frozen_lake_policy, frozen_lake_scores, frozen_lake_accumulated_scores = \
            frozen_lake_model.perform_MC_non_exploring_starts_control(print_info=print_q_table_and_policy)
        Utils.plot_running_average(method_name + ' - ' + frozen_lake_env.name, frozen_lake_scores,
                                   window=episodes//100, show=True)
        Utils.plot_accumulated_scores(method_name + ' - ' + frozen_lake_env.name, frozen_lake_accumulated_scores,
                                      show=True)
        frozen_lake_scores, frozen_lake_accumulated_scores = Utils.test_policy(frozen_lake_env, frozen_lake_policy)
        Utils.plot_running_average(method_name + ' - ' + frozen_lake_env.name, frozen_lake_scores,
                                   window=episodes//100, show=True)
        Utils.plot_accumulated_scores(method_name + ' - ' + frozen_lake_env.name, frozen_lake_accumulated_scores,
                                      show=True)

        # Blackjack:
        blackjack_env = Envs_DSS.Blackjack()
        blackjack_model = TabularMethods.MonteCarloModel(blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7)
        blackjack_policy, _, blackjack_accumulated_scores = \
            blackjack_model.perform_MC_non_exploring_starts_control(print_info=print_q_table_and_policy)
        Utils.plot_accumulated_scores(method_name + ' - ' + blackjack_env.name, blackjack_accumulated_scores, show=True)
        blackjack_accumulated_scores = Utils.test_policy(blackjack_env, blackjack_policy)
        Utils.plot_accumulated_scores(method_name + ' - ' + blackjack_env.name, blackjack_accumulated_scores, show=True)

    @staticmethod
    def test_off_policy_mc_control(episodes, print_q_table_and_policy):
        method_name = 'Off-policy MC Control'

        # Frozen Lake:
        frozen_lake_env = Envs_DSS.FrozenLake()
        frozen_lake_model = TabularMethods.MonteCarloModel(frozen_lake_env, episodes=episodes,
                                                           eps_max=frozen_lake_env.EPS_MIN)
        frozen_lake_policy, frozen_lake_scores, frozen_lake_accumulated_scores = \
            frozen_lake_model.perform_off_policy_MC_control(print_info=print_q_table_and_policy)
        Utils.plot_running_average(method_name + ' - ' + frozen_lake_env.name, frozen_lake_scores,
                                   window=episodes//100, show=True)
        Utils.plot_accumulated_scores(method_name + ' - ' + frozen_lake_env.name, frozen_lake_accumulated_scores,
                                      show=True)
        frozen_lake_scores, frozen_lake_accumulated_scores = Utils.test_policy(frozen_lake_env, frozen_lake_policy)
        Utils.plot_running_average(method_name + ' - ' + frozen_lake_env.name, frozen_lake_scores,
                                   window=episodes//100, show=True)
        Utils.plot_accumulated_scores(method_name + ' - ' + frozen_lake_env.name, frozen_lake_accumulated_scores,
                                      show=True)

        # Blackjack:
        blackjack_env = Envs_DSS.Blackjack()
        blackjack_model = TabularMethods.MonteCarloModel(blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7)
        blackjack_policy, _, blackjack_accumulated_scores = \
            blackjack_model.perform_off_policy_MC_control(print_info=print_q_table_and_policy)
        Utils.plot_accumulated_scores(method_name + ' - ' + blackjack_env.name, blackjack_accumulated_scores, show=True)
        blackjack_accumulated_scores = Utils.test_policy(blackjack_env, blackjack_policy)
        Utils.plot_accumulated_scores(method_name + ' - ' + blackjack_env.name, blackjack_accumulated_scores, show=True)

    @staticmethod
    def test_sarsa():
        method_name = 'SARSA'

        # Taxi:
        taxi_env = Envs_DSS.Taxi()
        taxi_model = TabularMethods.GeneralModel(taxi_env, episodes=10000, alpha=0.4)
        taxi_Q_table, taxi_scores = taxi_model.perform_sarsa()
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_scores, show=True)
        taxi_scores = Utils.test_q_table(taxi_env, taxi_Q_table)
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_scores, show=True)

        # Mountain Car:
        mountain_car_env = Envs_DSS.MountainCar()
        mountain_car_model = TabularMethods.GeneralModel(mountain_car_env, episodes=50000)
        mountain_car_Q_table, mountain_car_scores = mountain_car_model.perform_sarsa()
        Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name, mountain_car_scores, show=True)
        mountain_car_scores = Utils.test_q_table(mountain_car_env, mountain_car_Q_table)
        Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name, mountain_car_scores, show=True)

        # Cart Pole (Solved):
        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_model = TabularMethods.GeneralModel(cart_pole_env, episodes=50000)
        cart_pole_Q_table, cart_pole_scores = cart_pole_model.perform_sarsa()
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_scores, show=True)
        cart_pole_scores = Utils.test_q_table(cart_pole_env, cart_pole_Q_table)
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_scores, show=True)

        # Acrobot:
        acrobot_env = Envs_DSS.Acrobot()
        acrobot_model = TabularMethods.GeneralModel(acrobot_env, episodes=50000)
        acrobot_Q_table, acrobot_scores = acrobot_model.perform_sarsa()
        Utils.plot_running_average(method_name + ' - ' + acrobot_env.name, acrobot_scores, show=True)
        acrobot_scores = Utils.test_q_table(acrobot_env, acrobot_Q_table)
        Utils.plot_running_average(method_name + ' - ' + acrobot_env.name, acrobot_scores, show=True)

    @staticmethod
    def test_expected_sarsa():
        method_name = 'Expected SARSA'

        # Taxi:
        taxi_env = Envs_DSS.Taxi()
        taxi_model = TabularMethods.GeneralModel(taxi_env, episodes=10000, alpha=0.4)
        taxi_Q_table, taxi_scores = taxi_model.perform_expected_sarsa()
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_scores, show=True)
        taxi_scores = Utils.test_q_table(taxi_env, taxi_Q_table)
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_scores, show=True)

        # Mountain Car:
        mountain_car_env = Envs_DSS.MountainCar()
        mountain_car_model = TabularMethods.GeneralModel(mountain_car_env, episodes=50000)
        mountain_car_Q_table, mountain_car_scores = mountain_car_model.perform_expected_sarsa()
        Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name, mountain_car_scores, show=True)
        mountain_car_scores = Utils.test_q_table(mountain_car_env, mountain_car_Q_table)
        Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name, mountain_car_scores, show=True)

        # Cart Pole (Solved):
        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_model = TabularMethods.GeneralModel(cart_pole_env, episodes=50000)
        cart_pole_Q_table, cart_pole_scores = cart_pole_model.perform_expected_sarsa()
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_scores, show=True)
        cart_pole_scores = Utils.test_q_table(cart_pole_env, cart_pole_Q_table)
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_scores, show=True)

        # Acrobot:
        acrobot_env = Envs_DSS.Acrobot()
        acrobot_model = TabularMethods.GeneralModel(acrobot_env, episodes=50000)
        acrobot_Q_table, acrobot_scores = acrobot_model.perform_expected_sarsa()
        Utils.plot_running_average(method_name + ' - ' + acrobot_env.name, acrobot_scores, show=True)
        acrobot_scores = Utils.test_q_table(acrobot_env, acrobot_Q_table)
        Utils.plot_running_average(method_name + ' - ' + acrobot_env.name, acrobot_scores, show=True)

    @staticmethod
    def test_q_learning():
        method_name = 'Q-learning'

        # Taxi:
        taxi_env = Envs_DSS.Taxi()
        taxi_model = TabularMethods.GeneralModel(taxi_env, episodes=10000, alpha=0.4)
        taxi_Q_table, taxi_scores = taxi_model.perform_q_learning()
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_scores, show=True)
        taxi_scores = Utils.test_q_table(taxi_env, taxi_Q_table)
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_scores, show=True)

        # Mountain Car:
        mountain_car_env = Envs_DSS.MountainCar()
        mountain_car_model = TabularMethods.GeneralModel(mountain_car_env, episodes=50000)
        mountain_car_Q_table, mountain_car_scores = mountain_car_model.perform_q_learning()
        Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name, mountain_car_scores, show=True)
        mountain_car_scores = Utils.test_q_table(mountain_car_env, mountain_car_Q_table)
        Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name, mountain_car_scores, show=True)

        # Cart Pole:
        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_model = TabularMethods.GeneralModel(cart_pole_env, episodes=50000)
        cart_pole_Q_table, cart_pole_scores = cart_pole_model.perform_q_learning()
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_scores, show=True)
        cart_pole_scores = Utils.test_q_table(cart_pole_env, cart_pole_Q_table)
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_scores, show=True)

    @staticmethod
    def test_double_q_learning():
        method_name = 'Double Q-learning'

        # Taxi:
        taxi_env = Envs_DSS.Taxi()
        taxi_model = TabularMethods.GeneralModel(taxi_env, episodes=10000, alpha=0.4)
        taxi_Q1_table, taxi_Q2_table, taxi_scores = taxi_model.perform_double_q_learning()
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_scores, show=True)
        taxi_Q1_scores = Utils.test_q_table(taxi_env, taxi_Q1_table)
        taxi_Q2_scores = Utils.test_q_table(taxi_env, taxi_Q2_table)
        scores_list = [taxi_Q1_scores, taxi_Q2_scores]
        labels = ['Q1', 'Q2']
        Utils.plot_running_average_comparison(method_name + ' - ' + taxi_env.name, scores_list, labels, show=True)

        # Mountain Car:
        mountain_car_env = Envs_DSS.MountainCar()
        mountain_car_model = TabularMethods.GeneralModel(mountain_car_env, episodes=50000)
        mountain_car_Q1_table, mountain_car_Q2_table, mountain_car_scores = \
            mountain_car_model.perform_double_q_learning()
        Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name, mountain_car_scores, show=True)
        mountain_car_Q1_scores = Utils.test_q_table(mountain_car_env, mountain_car_Q1_table)
        mountain_car_Q2_scores = Utils.test_q_table(mountain_car_env, mountain_car_Q2_table)
        scores_list = [mountain_car_Q1_scores, mountain_car_Q2_scores]
        labels = ['Q1', 'Q2']
        Utils.plot_running_average_comparison(method_name + ' - ' + mountain_car_env.name, scores_list, labels, show=True)

        # Cart Pole:
        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_model = TabularMethods.GeneralModel(cart_pole_env, episodes=50000)
        cart_pole_Q1_table, cart_pole_Q2_table, cart_pole_scores = \
            cart_pole_model.perform_double_q_learning()
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_scores, show=True)
        cart_pole_Q1_scores = Utils.test_q_table(cart_pole_env, cart_pole_Q1_table)
        cart_pole_Q2_scores = Utils.test_q_table(cart_pole_env, cart_pole_Q2_table)
        scores_list = [cart_pole_Q1_scores, cart_pole_Q2_scores]
        labels = ['Q1', 'Q2']
        Utils.plot_running_average_comparison(method_name + ' - ' + cart_pole_env.name, scores_list, labels, show=True)


class EnvironmentsTesting:

    @staticmethod
    def test_environment(env, episodes, eps_max=1.0, eps_dec=None, alpha=0.1,
                         q_table_test_method=Utils.test_q_table,
                         policy_test_method=Utils.test_policy,
                         show_scores=True, show_accumulated_scores=True):

        labels = ['MC non-exploring starts', 'off-policy MC',
                  'SARSA', 'Expected SARSA', 'Q-learning', 'Double Q-learning']

        mc_model_01 = TabularMethods.MonteCarloModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
        policy_mc_01, scores_mc_01, accumulated_scores_mc_01 = mc_model_01.perform_MC_non_exploring_starts_control()

        mc_model_02 = TabularMethods.MonteCarloModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
        policy_mc_02, scores_mc_02, accumulated_scores_mc_02 = mc_model_02.perform_off_policy_MC_control()

        sarsa_model = TabularMethods.GeneralModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
        Q_table_sarsa, scores_sarsa, accumulated_scores_sarsa = sarsa_model.perform_sarsa()

        e_sarsa_model = TabularMethods.GeneralModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
        Q_table_e_sarsa, scores_e_sarsa, accumulated_scores_e_sarsa = e_sarsa_model.perform_expected_sarsa()

        q_l_model = TabularMethods.GeneralModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
        Q_table_q_l, scores_q_l, accumulated_scores_q_l = q_l_model.perform_q_learning()

        d_q_l_model = TabularMethods.GeneralModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
        Q1_table_d_q_l, Q2_table_d_q_l, scores_d_q_l, accumulated_scores_d_q_l = d_q_l_model.perform_double_q_learning()
        Q_table_d_q_l = {}
        for s in Q1_table_d_q_l:
            Q_table_d_q_l[s] = (Q1_table_d_q_l[s] + Q2_table_d_q_l[s]) / 2

        if show_scores:
            scores_list = [scores_mc_01, scores_mc_02, scores_sarsa, scores_e_sarsa, scores_q_l, scores_d_q_l]
            Utils.plot_running_average_comparison(env.name + ' - Training', scores_list, labels,  # window=episodes//100,
                                                  file_name=env.file_name + '-score-training')

        if show_accumulated_scores:
            accumulated_scores_list = [accumulated_scores_mc_01, accumulated_scores_mc_02,
                                       accumulated_scores_sarsa, accumulated_scores_e_sarsa,
                                       accumulated_scores_q_l, accumulated_scores_d_q_l]
            Utils.plot_accumulated_scores_comparison(env.name + ' - Training', accumulated_scores_list, labels,
                                                     file_name=env.file_name + '-accumulated-score-training')

        scores_mc_01, accumulated_scores_mc_01 = policy_test_method(env, policy_mc_01)
        scores_mc_02, accumulated_scores_mc_02 = policy_test_method(env, policy_mc_02)
        scores_sarsa, accumulated_scores_sarsa = q_table_test_method(env, Q_table_sarsa)
        scores_e_sarsa, accumulated_scores_e_sarsa = q_table_test_method(env, Q_table_e_sarsa)
        scores_q_l, accumulated_scores_q_l = q_table_test_method(env, Q_table_q_l)
        scores_d_q_l, accumulated_scores_d_q_l = q_table_test_method(env, Q_table_d_q_l)

        if show_scores:
            scores_list = [scores_mc_01, scores_mc_02, scores_sarsa, scores_e_sarsa, scores_q_l, scores_d_q_l]
            Utils.plot_running_average_comparison(env.name + ' - Test', scores_list, labels,  # window=episodes//100,
                                                  file_name=env.file_name + '-scores-test')

        if show_accumulated_scores:
            accumulated_scores_list = [accumulated_scores_mc_01, accumulated_scores_mc_02,
                                       accumulated_scores_sarsa, accumulated_scores_e_sarsa,
                                       accumulated_scores_q_l, accumulated_scores_d_q_l]
            Utils.plot_accumulated_scores_comparison(env.name + ' - Test', accumulated_scores_list, labels,
                                                     file_name=env.file_name + '-accumulated-scores-test')


def policy_evaluation_algorithms_test():
    AlgorithmsTesting.test_td0_policy_evaluation(10, print_v_table=True)
    AlgorithmsTesting.test_mc_policy_evaluation(10, print_v_table=True)


def learning_algorithms_test():
    AlgorithmsTesting.test_mc_non_exploring_starts_control(100000, print_q_table_and_policy=True)
    AlgorithmsTesting.test_off_policy_mc_control(100000, print_q_table_and_policy=False)
    AlgorithmsTesting.test_sarsa()
    AlgorithmsTesting.test_expected_sarsa()
    AlgorithmsTesting.test_q_learning()
    AlgorithmsTesting.test_double_q_learning()


def environments_test():
    # EnvironmentsTesting.test_environment(Envs_DSS.FrozenLake(), episodes=100000, eps_max=0.1, eps_dec=None)
    # EnvironmentsTesting.test_environment(Envs_DSS.Blackjack(), episodes=100000, eps_max=0.05, eps_dec=1e-7)
    # EnvironmentsTesting.test_environment(Envs_DSS.Taxi(), episodes=10000, alpha=0.4)
    # EnvironmentsTesting.test_environment(Envs_DSS.MountainCar(), episodes=50000)
    EnvironmentsTesting.test_environment(Envs_DSS.CartPole(), episodes=50000)
    # EnvironmentsTesting.test_environment(Envs_DSS.Acrobot(), episodes=50000)


if __name__ == '__main__':
    # policy_evaluation_algorithms_test()
    # learning_algorithms_test()
    environments_test()
