from utils import Utils
from tabular_methods.envs_dss import Envs_DSS
from tabular_methods.rl_tabular import TabularMethods


class AlgorithmsTesting:

    @staticmethod
    def test_td_zero(episodes):
        method_name = 'TD(0)'

        mountain_car_env_single_state_space = Envs_DSS.MountainCar(0)
        mountain_car_td_zero_model = TabularMethods.TdZeroModel(mountain_car_env_single_state_space, episodes=episodes)
        mountain_car_total_rewards_td_zero_model = mountain_car_td_zero_model.perform()
        # Utils.plot_running_average(method_name + ' - ' + mountain_car_env_single_state_space.name,
        #                            mountain_car_total_rewards_td_zero_model)

        cart_pole_env_single_state_space = Envs_DSS.CartPole(2)
        cart_pole_td_zero_model = TabularMethods.TdZeroModel(cart_pole_env_single_state_space, episodes=episodes)
        cart_pole_total_rewards_td_zero_model = cart_pole_td_zero_model.perform()
        # Utils.plot_running_average(method_name + ' - ' + cart_pole_env_single_state_space.name,
        #                            cart_pole_total_rewards_td_zero_model)

        total_rewards_list = [mountain_car_total_rewards_td_zero_model, cart_pole_total_rewards_td_zero_model]
        labels = [mountain_car_env_single_state_space.name, cart_pole_env_single_state_space.name]
        Utils.plot_running_average_comparison(method_name, total_rewards_list, labels)

    @staticmethod
    def test_mc_policy_evaluation(episodes):
        frozen_lake_env = Envs_DSS.FrozenLake()
        frozen_lake_monte_carlo_model = TabularMethods.MonteCarloModel(
            frozen_lake_env, episodes=episodes, eps_dec=2/episodes
        )
        frozen_lake_monte_carlo_model.perform_MC_policy_evaluation(print_info=True)

        mountain_car_env_single_state_space = Envs_DSS.MountainCar(0)
        mountain_car_monte_carlo_model = TabularMethods.MonteCarloModel(
            mountain_car_env_single_state_space, episodes=episodes, eps_dec=2/episodes
        )
        mountain_car_monte_carlo_model.perform_MC_policy_evaluation(print_info=True)

    @staticmethod
    def test_mc_non_exploring_starts_control(episodes):
        method_name = 'MC non-exploring starts'

        frozen_lake_env = Envs_DSS.FrozenLake()
        frozen_lake_monte_carlo_model = TabularMethods.MonteCarloModel(
            frozen_lake_env, episodes=episodes, eps_max=frozen_lake_env.EPS_MIN
        )
        frozen_lake_total_rewards_mc_without_es, frozen_lake_accumulated_rewards_mc_without_es = \
            frozen_lake_monte_carlo_model.perform_MC_non_exploring_starts_control(print_info=True)
        # Utils.plot_running_average(method_name + ' - ' + frozen_lake_env.name,
        #                            frozen_lake_total_rewards_mc_without_es)
        Utils.plot_accumulated_rewards(method_name + ' - ' + frozen_lake_env.name,
                                       frozen_lake_accumulated_rewards_mc_without_es)

        blackjack_env = Envs_DSS.Blackjack()
        blackjack_monte_carlo_model = TabularMethods.MonteCarloModel(
            blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7
        )
        blackjack_total_rewards_mc_without_es, blackjack_accumulated_rewards_mc_without_es = \
            blackjack_monte_carlo_model.perform_MC_non_exploring_starts_control(print_info=True)
        # Utils.plot_running_average(method_name + ' - ' + blackjack_env.name,
        #                            blackjack_total_rewards_mc_without_es)
        Utils.plot_accumulated_rewards(method_name + ' - ' + blackjack_env.name,
                                       blackjack_accumulated_rewards_mc_without_es)

        # total_rewards_list = [frozen_lake_total_rewards_mc_without_es,
        #                       blackjack_total_rewards_mc_without_es]
        accumulated_rewards_list = [frozen_lake_accumulated_rewards_mc_without_es,
                                    blackjack_accumulated_rewards_mc_without_es]
        labels = [frozen_lake_env.name, blackjack_env.name]
        # Utils.plot_running_average_comparison(method_name, total_rewards_list, labels)
        Utils.plot_accumulated_rewards_comparison(method_name, accumulated_rewards_list, labels)

    @staticmethod
    def test_off_policy_mc_control(episodes):
        method_name = 'Off-policy MC Control'

        blackjack_env = Envs_DSS.Blackjack()
        blackjack_monte_carlo_model = TabularMethods.MonteCarloModel(
            blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7
        )
        blackjack_total_rewards_off_policy_mc, blackjack_accumulated_rewards_off_policy_mc = \
            blackjack_monte_carlo_model.perform_off_policy_MC_control(print_info=True)
        # Utils.plot_running_average(method_name + ' - ' + blackjack_env.name,
        #                            blackjack_total_rewards_off_policy_mc)
        Utils.plot_accumulated_rewards(method_name + ' - ' + blackjack_env.name,
                                       blackjack_accumulated_rewards_off_policy_mc)

    @staticmethod
    def test_sarsa(episodes):
        method_name = 'SARSA'

        taxi_env = Envs_DSS.Taxi()
        taxi_sarsa_model = TabularMethods.GeneralModel(taxi_env, alpha=0.4, episodes=episodes, eps_dec=2/episodes)
        taxi_total_rewards_sarsa_model = taxi_sarsa_model.perform_sarsa()
        # Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_total_rewards_sarsa_model)

        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_sarsa_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes, eps_dec=2/episodes)
        cart_pole_total_rewards_sarsa_model = cart_pole_sarsa_model.perform_sarsa()
        # Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_total_rewards_sarsa_model)

        acrobot_env = Envs_DSS.Acrobot()
        acrobot_sarsa_model = TabularMethods.GeneralModel(acrobot_env, episodes=episodes, eps_dec=2/episodes)
        acrobot_total_rewards_sarsa_model = acrobot_sarsa_model.perform_sarsa()
        # Utils.plot_running_average(method_name + ' - ' + acrobot_env.name, acrobot_total_rewards_sarsa_model)

        total_rewards_list = [taxi_total_rewards_sarsa_model, cart_pole_total_rewards_sarsa_model,
                              acrobot_total_rewards_sarsa_model]
        labels = [taxi_env.name, cart_pole_env.name, acrobot_env.name]
        Utils.plot_running_average_comparison(method_name, total_rewards_list, labels)

    @staticmethod
    def test_expected_sarsa(episodes):
        method_name = 'Expected SARSA'

        taxi_env = Envs_DSS.Taxi()
        taxi_expected_sarsa_model = TabularMethods.GeneralModel(
            taxi_env, alpha=0.4, episodes=episodes, eps_dec=2 / episodes
        )
        taxi_total_rewards_expected_sarsa_model = taxi_expected_sarsa_model.perform_expected_sarsa()
        Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_total_rewards_expected_sarsa_model)

    @staticmethod
    def test_q_learning(episodes):
        method_name = 'Q-learning'

        taxi_env = Envs_DSS.Taxi()
        taxi_q_learning_model = TabularMethods.GeneralModel(taxi_env, alpha=0.4, episodes=episodes)
        taxi_total_rewards_q_learning_model = taxi_q_learning_model.perform_q_learning()
        # Utils.plot_running_average(method_name + ' - ' + taxi_env.name, taxi_total_rewards_q_learning_model)

        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_q_learning_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes)
        cart_pole_total_rewards_q_learning_model = cart_pole_q_learning_model.perform_q_learning()
        # Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name, cart_pole_total_rewards_q_learning_model)

        mountain_car_env = Envs_DSS.MountainCar()
        mountain_car_q_learning_model = TabularMethods.GeneralModel(mountain_car_env, episodes=episodes)
        mountain_car_total_rewards_q_learning_model = mountain_car_q_learning_model.perform_q_learning()
        # Utils.plot_running_average(method_name + ' - ' + mountain_car_env.name,
        #                            mountain_car_total_rewards_q_learning_model)

        total_rewards_list = [taxi_total_rewards_q_learning_model, cart_pole_total_rewards_q_learning_model,
                              mountain_car_total_rewards_q_learning_model]
        labels = [taxi_env.name, cart_pole_env.name, mountain_car_env.name]
        Utils.plot_running_average_comparison(method_name, total_rewards_list, labels)

    @staticmethod
    def test_double_q_learning(episodes):
        method_name = 'Double Q-learning'

        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_double_q_learning_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes)
        cart_pole_total_rewards_double_q_learning_model = cart_pole_double_q_learning_model.perform_double_q_learning()
        Utils.plot_running_average(method_name + ' - ' + cart_pole_env.name,
                                   cart_pole_total_rewards_double_q_learning_model)


class EnvironmentsTesting:

    @staticmethod
    def test_frozen_lake(episodes):
        frozen_lake_env = Envs_DSS.FrozenLake()

        monte_carlo_model_01 = TabularMethods.MonteCarloModel(
            frozen_lake_env, episodes=episodes, eps_max=frozen_lake_env.EPS_MIN
        )
        total_rewards_mc_without_es, accumulated_rewards_mc_without_es = \
            monte_carlo_model_01.perform_MC_non_exploring_starts_control(print_info=True)
        # Utils.plot_running_average('Frozen Lake', total_rewards_mc_without_es)          # less preferred
        Utils.plot_accumulated_rewards('Frozen Lake', accumulated_rewards_mc_without_es)  # better

        monte_carlo_model_02 = TabularMethods.MonteCarloModel(frozen_lake_env, episodes=episodes, eps_dec=2/episodes)
        monte_carlo_model_02.perform_MC_policy_evaluation(print_info=True)

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

        monte_carlo_model_01 = TabularMethods.MonteCarloModel(
            blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7
        )
        total_rewards_mc_without_es, accumulated_rewards_mc_without_es = \
            monte_carlo_model_01.perform_MC_non_exploring_starts_control(print_info=True)

        monte_carlo_model_02 = TabularMethods.MonteCarloModel(
            blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7
        )
        total_rewards_off_policy_mc, accumulated_rewards_off_policy_mc = \
            monte_carlo_model_02.perform_off_policy_MC_control(print_info=True)

        # less preferred:
        # total_rewards_list = [total_rewards_mc_without_es, total_rewards_off_policy_mc]
        # labels=['MC non-exploring starts', 'off policy MC']
        # Utils.plot_running_average_comparison('Blackjack', total_rewards_list, labels)

        # better:
        accumulated_rewards_list = [accumulated_rewards_mc_without_es, accumulated_rewards_off_policy_mc]
        labels = ['MC non-exploring starts', 'off policy MC']
        Utils.plot_accumulated_rewards_comparison('Blackjack', accumulated_rewards_list, labels)

    @staticmethod
    def test_cart_pole(episodes):
        cart_pole_env = Envs_DSS.CartPole()
        cart_pole_env_single_state_space = Envs_DSS.CartPole(2)

        td_zero_model = TabularMethods.TdZeroModel(cart_pole_env_single_state_space, episodes=episodes)
        total_rewards_td_zero_model = td_zero_model.perform()

        sarsa_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes, eps_dec=2/episodes)
        total_rewards_sarsa_model = sarsa_model.perform_sarsa()

        q_learning_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes)
        total_rewards_q_learning_model = q_learning_model.perform_q_learning()

        double_q_learning_model = TabularMethods.GeneralModel(cart_pole_env, episodes=episodes)
        total_rewards_double_q_learning_model = double_q_learning_model.perform_double_q_learning()

        total_rewards_list = [total_rewards_td_zero_model, total_rewards_sarsa_model,
                              total_rewards_q_learning_model, total_rewards_double_q_learning_model]
        labels = ['TD(0)', 'SARSA', 'Q-learning', 'Double Q-learning']
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
        mountain_car_env_single_state_space = Envs_DSS.MountainCar(0)

        monte_carlo_model = TabularMethods.MonteCarloModel(mountain_car_env_single_state_space, episodes=episodes,
                                                           eps_dec=2/episodes)
        monte_carlo_model.perform_MC_policy_evaluation()

        td_zero_model = TabularMethods.TdZeroModel(mountain_car_env_single_state_space, episodes=episodes)
        total_rewards_td_zero_model = td_zero_model.perform()

        q_learning_model = TabularMethods.GeneralModel(mountain_car_env, episodes=episodes)
        total_rewards_q_learning_model = q_learning_model.perform_q_learning()

        total_rewards_list = [total_rewards_td_zero_model, total_rewards_q_learning_model]
        labels = ['TD(0)', 'Q-learning']
        Utils.plot_running_average_comparison('Mountain Car', total_rewards_list, labels)


def test_by_algorithm():
    AlgorithmsTesting.test_td_zero(1000)
    # RLTabularTesting.test_MC_policy_evaluation(1000)
    AlgorithmsTesting.test_mc_non_exploring_starts_control(1000)
    AlgorithmsTesting.test_off_policy_mc_control(1000)
    AlgorithmsTesting.test_sarsa(1000)
    AlgorithmsTesting.test_expected_sarsa(1000)
    AlgorithmsTesting.test_q_learning(1000)
    AlgorithmsTesting.test_double_q_learning(1000)


def test_by_environment():
    EnvironmentsTesting.test_frozen_lake(1000)  # 100000
    EnvironmentsTesting.test_taxi(1000)         # 2000, 10000
    EnvironmentsTesting.test_blackjack(1000)    # 100000
    EnvironmentsTesting.test_cart_pole(1000)    # 50000
    EnvironmentsTesting.test_acrobot(1000)
    EnvironmentsTesting.test_mountain_car(1000)  # 50000


if __name__ == '__main__':
    test_by_algorithm()
    # test_by_environment()
