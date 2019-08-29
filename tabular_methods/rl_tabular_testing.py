from utils import Utils
from tabular_methods.envs_dss import Envs_DSS
from tabular_methods.rl_tabular import TabularMethods


class AlgorithmsTesting:

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
    def test_mc_policy_evaluation(episodes, print_v_table):
        # Mountain Car:
        car_vel_bin_num = 34  # ~100% success rate (32 rarely loses)
        mountain_car_env = Envs_DSS.MountainCar(Envs_DSS.MountainCar.CAR_VEL, car_vel_bin_num=car_vel_bin_num)
        mountain_car_model = TabularMethods.MonteCarloModel(mountain_car_env, episodes=episodes, eps_dec=2/episodes)
        # possible actions: backward (0), none (1), forward (2)
        mountain_car_policy = lambda velocity_state: 0 if velocity_state < (car_vel_bin_num//2) else 2
        mountain_car_model.perform_MC_policy_evaluation(mountain_car_policy, print_info=print_v_table)

        # Cart Pole:
        pole_theta_bin_num = 10
        cart_pole_env = Envs_DSS.CartPole(Envs_DSS.CartPole.POLE_THETA, pole_theta_bin_num=pole_theta_bin_num)
        cart_pole_model = TabularMethods.MonteCarloModel(cart_pole_env, episodes=episodes, eps_dec=2/episodes)
        # possible actions: left (0), right (1)
        cart_pole_policy = lambda theta_state: 0 if theta_state < (pole_theta_bin_num // 2) else 1
        cart_pole_model.perform_MC_policy_evaluation(cart_pole_policy, print_info=print_v_table)

        # Frozen Lake:
        frozen_lake_env = Envs_DSS.FrozenLake()
        frozen_lake_model = TabularMethods.MonteCarloModel(frozen_lake_env, episodes=episodes, eps_dec=2/episodes)
        frozen_lake_policy = lambda s: frozen_lake_env.env.action_space.sample()  # random policy
        frozen_lake_model.perform_MC_policy_evaluation(frozen_lake_policy, print_info=print_v_table)

    @staticmethod
    def test_mc_non_exploring_starts_control(episodes, print_q_table_and_policy):
        method_name = 'MC non-exploring starts'

        # Frozen Lake:
        frozen_lake_env = Envs_DSS.FrozenLake()
        frozen_lake_model = TabularMethods.MonteCarloModel(
            frozen_lake_env, episodes=episodes, eps_max=frozen_lake_env.EPS_MIN)
        frozen_lake_policy, frozen_lake_total_rewards, frozen_lake_accumulated_rewards = \
            frozen_lake_model.perform_MC_non_exploring_starts_control(print_info=print_q_table_and_policy)
        Utils.plot_running_average(method_name + ' - ' + frozen_lake_env.name,
                                   frozen_lake_total_rewards, window=episodes//100)
        Utils.plot_accumulated_rewards(method_name + ' - ' + frozen_lake_env.name, frozen_lake_accumulated_rewards)
        frozen_lake_env.test_policy(frozen_lake_policy)

        # Blackjack:
        blackjack_env = Envs_DSS.Blackjack()
        blackjack_model = TabularMethods.MonteCarloModel( blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7)
        blackjack_policy, _, blackjack_accumulated_rewards = \
            blackjack_model.perform_MC_non_exploring_starts_control(print_info=print_q_table_and_policy)
        Utils.plot_accumulated_rewards(method_name + ' - ' + blackjack_env.name, blackjack_accumulated_rewards)
        blackjack_env.test_policy(blackjack_policy)

    @staticmethod
    def test_off_policy_mc_control(episodes, print_q_table_and_policy):
        method_name = 'Off-policy MC Control'

        # Blackjack:
        blackjack_env = Envs_DSS.Blackjack()
        blackjack_model = TabularMethods.MonteCarloModel(blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7)
        _, _, blackjack_accumulated_rewards = \
            blackjack_model.perform_off_policy_MC_control(print_info=print_q_table_and_policy)
        Utils.plot_accumulated_rewards(method_name + ' - ' + blackjack_env.name, blackjack_accumulated_rewards)

    @staticmethod
    def test_sarsa():
        method_name = 'SARSA'

        # Taxi:
        taxi_env = Envs_DSS.Taxi()
        taxi_model = TabularMethods.GeneralModel(taxi_env, alpha=0.4, episodes=1000)
        taxi_Q_table, taxi_total_rewards = taxi_model.perform_sarsa()
        # Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_total_rewards, window=100, show=True)
        taxi_total_rewards = Utils.test_q_table(taxi_env, taxi_Q_table)
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_total_rewards, show=True)

        # Cart Pole:
        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_model = TabularMethods.GeneralModel(cart_pole_env, episodes=50000)
        cart_pole_Q_table, cart_pole_total_rewards = cart_pole_model.perform_sarsa(record=True)
        # Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_total_rewards, window=100, show=True)
        cart_pole_total_rewards = Utils.test_q_table(cart_pole_env, cart_pole_Q_table)
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_total_rewards, show=True)

        # Acrobot:
        acrobot_env = Envs_DSS.Acrobot()
        acrobot_model = TabularMethods.GeneralModel(acrobot_env, episodes=2500)
        acrobot_Q_table, acrobot_total_rewards = acrobot_model.perform_sarsa()
        # Utils.plot_running_average(method_name + ' - ' + acrobot_env.name, acrobot_total_rewards)
        # Utils.test_q_table(acrobot_env, acrobot_Q_table)

        total_rewards_list = [taxi_total_rewards, cart_pole_total_rewards, acrobot_total_rewards]
        labels = [taxi_env.name, cart_pole_env.name, acrobot_env.name]
        Utils.plot_running_average_comparison(method_name, total_rewards_list, labels, show=True)  # window=episodes//10,

    @staticmethod
    def test_expected_sarsa(episodes):
        method_name = 'Expected SARSA'

        # Taxi:
        taxi_env = Envs_DSS.Taxi()
        taxi_model = TabularMethods.GeneralModel(taxi_env, alpha=0.4, episodes=episodes, eps_dec=2/episodes)
        taxi_Q_table, taxi_total_rewards = taxi_model.perform_expected_sarsa()
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_total_rewards)

    @staticmethod
    def test_q_learning(episodes):
        method_name = 'Q-learning'

        # Taxi:
        taxi_env = Envs_DSS.Taxi()
        taxi_model = TabularMethods.GeneralModel(taxi_env, alpha=0.4, episodes=episodes)
        taxi_Q_table, taxi_total_rewards = taxi_model.perform_q_learning()
        # Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_total_rewards)

        # Cart Pole:
        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes)
        cart_pole_Q_table, cart_pole_total_rewards = cart_pole_model.perform_q_learning()
        # Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_total_rewards)

        # Mountain Car:
        mountain_car_env = Envs_DSS.MountainCar()
        mountain_car_model = TabularMethods.GeneralModel(mountain_car_env, episodes=episodes)
        mountain_car_Q_table, mountain_car_total_rewards = mountain_car_model.perform_q_learning()
        # Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name,
        #                            mountain_car_total_rewards)

        total_rewards_list = [taxi_total_rewards, cart_pole_total_rewards, mountain_car_total_rewards]
        labels = [taxi_env.name, cart_pole_env.name, mountain_car_env.name]
        Utils.plot_running_average_comparison(method_name, total_rewards_list, labels)

    @staticmethod
    def test_double_q_learning(episodes):
        method_name = 'Double Q-learning'

        # Cart Pole:
        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_double_q_learning_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes)
        cart_pole_total_rewards_double_q_learning_model = cart_pole_double_q_learning_model.perform_double_q_learning()
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name,
                                   cart_pole_total_rewards_double_q_learning_model)


class EnvironmentsTesting:

    @staticmethod
    def test_frozen_lake(episodes):
        frozen_lake_env = Envs_DSS.FrozenLake()

        mc_model_01 = TabularMethods.MonteCarloModel(
            frozen_lake_env, episodes=episodes, eps_max=frozen_lake_env.EPS_MIN
        )
        total_rewards_mc_without_es, accumulated_rewards_mc_without_es = \
            mc_model_01.perform_MC_non_exploring_starts_control(print_info=True)
        # Utils.plot_running_average('Frozen Lake', total_rewards_mc_without_es)          # less preferred
        Utils.plot_accumulated_rewards('Frozen Lake', accumulated_rewards_mc_without_es)  # better

    @staticmethod
    def test_taxi(episodes):
        taxi_env = Envs_DSS.Taxi()

        alpha = 0.4

        sarsa_model = TabularMethods.GeneralModel(taxi_env, alpha=alpha, episodes=episodes, eps_dec=2/episodes)
        total_rewards_sarsa_model = sarsa_model.perform_sarsa()

        expected_sarsa_model = TabularMethods.GeneralModel(taxi_env, alpha=alpha, episodes=episodes, eps_dec=2/episodes)
        total_rewards_expected_sarsa_model = expected_sarsa_model.perform_expected_sarsa()

        q_learning_model = TabularMethods.GeneralModel(taxi_env, alpha=alpha, episodes=episodes)
        total_rewards_q_learning_model = q_learning_model.perform_q_learning()

        total_rewards_list = [total_rewards_sarsa_model, total_rewards_expected_sarsa_model,
                              total_rewards_q_learning_model]
        labels = ['SARSA', 'expected SARSA', 'Q-learning']
        Utils.plot_running_average_comparison('Taxi', total_rewards_list, labels)

    @staticmethod
    def test_blackjack(episodes):
        blackjack_env = Envs_DSS.Blackjack()

        mc_model_01 = TabularMethods.MonteCarloModel(blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7)
        _, accumulated_rewards_mc_without_es = \
            mc_model_01.perform_MC_non_exploring_starts_control(print_info=True)

        mc_model_02 = TabularMethods.MonteCarloModel(blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7)
        _, accumulated_rewards_off_policy_mc = \
            mc_model_02.perform_off_policy_MC_control(print_info=True)

        accumulated_rewards_list = [accumulated_rewards_mc_without_es, accumulated_rewards_off_policy_mc]
        labels = ['MC non-exploring starts', 'off policy MC']
        Utils.plot_accumulated_rewards_comparison('Blackjack', accumulated_rewards_list, labels)

    @staticmethod
    def test_cart_pole(episodes):
        cart_pole_env = Envs_DSS.CartPole()

        sarsa_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes, eps_dec=2/episodes)
        total_rewards_sarsa_model = sarsa_model.perform_sarsa()

        q_learning_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes)
        total_rewards_q_learning_model = q_learning_model.perform_q_learning()

        double_q_learning_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes)
        total_rewards_double_q_learning_model = double_q_learning_model.perform_double_q_learning()

        total_rewards_list = [total_rewards_sarsa_model,
                              total_rewards_q_learning_model, total_rewards_double_q_learning_model]
        labels = ['SARSA', 'Q-learning', 'Double Q-learning']
        Utils.plot_running_average_comparison('Cart Pole', total_rewards_list, labels)

    @staticmethod
    def test_acrobot(episodes):
        acrobot_env = Envs_DSS.Acrobot()

        sarsa_model = TabularMethods.GeneralModel(acrobot_env, episodes=episodes, eps_dec=2/episodes)
        total_rewards_sarsa_model = sarsa_model.perform_sarsa()
        Utils.plot_running_average('Acrobot', total_rewards_sarsa_model)

    @staticmethod
    def test_mountain_car(episodes):
        mountain_car_env = Envs_DSS.MountainCar()

        q_learning_model = TabularMethods.GeneralModel(mountain_car_env, episodes=episodes)
        total_rewards_q_learning_model = q_learning_model.perform_q_learning()

        total_rewards_list = [total_rewards_q_learning_model]
        labels = ['Q-learning']
        Utils.plot_running_average_comparison('Mountain Car', total_rewards_list, labels)


def policy_evaluation_algorithms_test():
    AlgorithmsTesting.test_td0_policy_evaluation(10, print_v_table=True)
    AlgorithmsTesting.test_mc_policy_evaluation(10, print_v_table=True)


def learning_algorithms_test():
    AlgorithmsTesting.test_mc_non_exploring_starts_control(100000, print_q_table_and_policy=True)
    AlgorithmsTesting.test_off_policy_mc_control(100000, print_q_table_and_policy=True)

    AlgorithmsTesting.test_sarsa()
    AlgorithmsTesting.test_expected_sarsa(1000)
    AlgorithmsTesting.test_q_learning(1000)
    AlgorithmsTesting.test_double_q_learning(1000)


def environments_test():
    EnvironmentsTesting.test_frozen_lake(100000)
    EnvironmentsTesting.test_taxi(1000)             # 2000, 3000, 10000
    EnvironmentsTesting.test_blackjack(100000)
    EnvironmentsTesting.test_cart_pole(1000)        # 50000
    EnvironmentsTesting.test_acrobot(1000)          # 100
    EnvironmentsTesting.test_mountain_car(1000)     # 50000


if __name__ == '__main__':
    # policy_evaluation_algorithms_test()
    learning_algorithms_test()
    # environments_test()
