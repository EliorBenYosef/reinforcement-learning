from utils import Utils
from tabular_methods.envs_dss import Envs_DSS
from tabular_methods.rl_tabular import TabularMethods


class RLTabularTesting:

    @staticmethod
    def test_frozen_lake(episodes):
        frozen_lake_env = Envs_DSS.FrozenLake()

        monte_carlo_model_01 = TabularMethods.MonteCarloModel(
            frozen_lake_env, episodes=episodes,
            eps_max=0.1, eps_min=0.1
        )
        total_rewards_mc_without_es, accumulated_rewards_mc_without_es = \
            monte_carlo_model_01.perform_MC_non_exploring_starts_control(print_info=True)
        # Utils.plot_running_average('Frozen Lake', total_rewards_mc_without_es)          # less preferred
        Utils.plot_accumulated_rewards('Frozen Lake', accumulated_rewards_mc_without_es)  # better

        monte_carlo_model_02 = TabularMethods.MonteCarloModel(
            frozen_lake_env, episodes=episodes,
            eps_dec=2/episodes, eps_dec_type=Utils.EPS_DEC_LINEAR
        )
        monte_carlo_model_02.perform_MC_policy_evaluation(print_info=True)

    @staticmethod
    def test_taxi(episodes):
        taxi_env = Envs_DSS.Taxi()

        alpha = 0.4

        eps_max = 1.0
        eps_min = 0.0  # deterministic environment

        sarsa_model = TabularMethods.GeneralModel(
            taxi_env, alpha=alpha, episodes=episodes,
            eps_max=eps_max, eps_min=eps_min, eps_dec=2/episodes, eps_dec_type=Utils.EPS_DEC_LINEAR
        )
        total_rewards_sarsa_model = sarsa_model.perform_sarsa()

        expected_sarsa_model = TabularMethods.GeneralModel(
            taxi_env, alpha=alpha, episodes=episodes,
            eps_max=eps_max, eps_min=eps_min, eps_dec=2/episodes, eps_dec_type=Utils.EPS_DEC_LINEAR
        )
        total_rewards_expected_sarsa_model = expected_sarsa_model.perform_expected_sarsa()

        q_learning_model = TabularMethods.GeneralModel(taxi_env, alpha=alpha, episodes=episodes,
                                                       eps_max=eps_max, eps_min=eps_min)
        total_rewards_q_learning_model = q_learning_model.perform_q_learning()

        total_rewards_list = [total_rewards_sarsa_model, total_rewards_expected_sarsa_model,
                              total_rewards_q_learning_model]
        labels = ['SARSA', 'expected SARSA', 'Q-learning']
        Utils.plot_running_average_comparison('Taxi', total_rewards_list, labels)

    @staticmethod
    def test_blackjack(episodes):
        blackjack_env = Envs_DSS.Blackjack()

        monte_carlo_model_01 = TabularMethods.MonteCarloModel(
            blackjack_env, episodes=episodes,
            eps_max=0.05, eps_dec=1e-7, eps_dec_type=Utils.EPS_DEC_LINEAR
        )
        total_rewards_mc_without_es, accumulated_rewards_mc_without_es = \
            monte_carlo_model_01.perform_MC_non_exploring_starts_control(print_info=True)

        monte_carlo_model_02 = TabularMethods.MonteCarloModel(
            blackjack_env, episodes=episodes,
            eps_max=0.05, eps_dec=1e-7, eps_dec_type=Utils.EPS_DEC_LINEAR
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

        sarsa_model = TabularMethods.GeneralModel(
            cart_pole_env, episodes=episodes,
            eps_dec=2/episodes, eps_dec_type=Utils.EPS_DEC_LINEAR
        )
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

        eps_max = 1.0
        eps_min = 0.0  # deterministic environment

        sarsa_model = TabularMethods.GeneralModel(
            acrobot_env, episodes=episodes,
            eps_max=eps_max, eps_min=eps_min, eps_dec=2/episodes, eps_dec_type=Utils.EPS_DEC_LINEAR
        )
        total_rewards_sarsa_model = sarsa_model.perform_sarsa()
        Utils.plot_running_average('Acrobot', total_rewards_sarsa_model)

    @staticmethod
    def test_mountain_car(episodes):
        mountain_car_env = Envs_DSS.MountainCar()
        mountain_car_env_single_state_space = Envs_DSS.MountainCar(0)

        monte_carlo_model = TabularMethods.MonteCarloModel(
            mountain_car_env_single_state_space, episodes=episodes, gamma=1.0,
            eps_dec=2/episodes, eps_dec_type=Utils.EPS_DEC_LINEAR
        )
        monte_carlo_model.perform_MC_policy_evaluation()

        td_zero_model = TabularMethods.TdZeroModel(
            mountain_car_env_single_state_space, episodes=episodes, gamma=1.0
        )
        total_rewards_td_zero_model = td_zero_model.perform()

        q_learning_model = TabularMethods.GeneralModel(
            mountain_car_env, episodes=episodes,
            eps_max=0.0, eps_min=0.01
        )
        total_rewards_q_learning_model = q_learning_model.perform_q_learning()

        total_rewards_list = [total_rewards_td_zero_model, total_rewards_q_learning_model]
        labels = ['TD(0)', 'Q-learning']
        Utils.plot_running_average_comparison('Mountain Car', total_rewards_list, labels)


if __name__ == '__main__':
    RLTabularTesting.test_frozen_lake(1000)  # 100000
    RLTabularTesting.test_taxi(1000)         # 2000, 10000
    RLTabularTesting.test_blackjack(1000)    # 100000
    RLTabularTesting.test_cart_pole(1000)    # 50000
    RLTabularTesting.test_acrobot(1000)
    RLTabularTesting.test_mountain_car(1000)  # 50000
