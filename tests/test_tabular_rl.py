from numpy.random import seed
seed(28)

from reinforcement_learning.tabular_RL.envs_dss import FrozenLake, Taxi, Blackjack, MountainCar, CartPole, Acrobot
from reinforcement_learning.tabular_RL.algorithms.monte_carlo import MCPredictionModel, MCControlModel
from reinforcement_learning.tabular_RL.algorithms.td_zero import TD0PredictionModel, TD0ControlModel
from reinforcement_learning.tabular_RL.utils import run_q_table, run_policy_table
from reinforcement_learning.utils.plotter import plot_running_average, plot_accumulated_scores, \
    plot_running_average_comparison, plot_accumulated_scores_comparison


# Prediction (policy evaluation algorithms)

def test_mc_policy_evaluation(episodes=5, print_v_table=False):  # episodes=10
    """
    Performs MC policy evaluation on multiple environments (separately).
    """
    # Mountain Car:
    car_vel_bin_num = 34  # ~100% success rate (32 rarely loses)
    mc_env = MountainCar(car_vel_bin_num, single_state_space=MountainCar.CAR_VEL)
    mc_model = MCPredictionModel(mc_env, episodes)
    mc_model.perform_mc_policy_evaluation(
        policy=lambda velocity_state: 0 if velocity_state < (car_vel_bin_num // 2) else 2,  # mc_policy
        print_info=print_v_table)

    # Cart Pole:
    pole_theta_bin_num = 10
    cp_env = CartPole(pole_theta_bin_num, single_state_space=CartPole.POLE_THETA)
    cp_model = MCPredictionModel(cp_env, episodes)
    cp_model.perform_mc_policy_evaluation(
        policy=lambda theta_state: 0 if theta_state < (pole_theta_bin_num // 2) else 1,  # cp_policy
        print_info=print_v_table)

    # Frozen Lake:
    fl_env = FrozenLake()
    fl_model = MCPredictionModel(fl_env, episodes)
    fl_model.perform_mc_policy_evaluation(
        policy=lambda s: fl_env.env.action_space.sample(),  # fl_policy - random policy
        print_info=print_v_table)


def test_td0_policy_evaluation(episodes=5, print_v_table=False):  # episodes=10
    """
    Performs TD0 policy evaluation on multiple environments (separately).
    """
    # Mountain Car:
    car_vel_bin_num = 34  # ~100% success rate (32 rarely loses)
    mc_env = MountainCar(car_vel_bin_num, single_state_space=MountainCar.CAR_VEL)
    mc_model = TD0PredictionModel(mc_env, episodes)
    mc_model.perform_td0_policy_evaluation(
        policy=lambda velocity_state: 0 if velocity_state < (car_vel_bin_num // 2) else 2,  # mc_policy
        print_info=print_v_table)

    # Cart Pole:
    pole_theta_bin_num = 10
    cp_env = CartPole(pole_theta_bin_num, single_state_space=CartPole.POLE_THETA)
    cp_model = TD0PredictionModel(cp_env, episodes)
    cp_model.perform_td0_policy_evaluation(
        policy=lambda theta_state: 0 if theta_state < (pole_theta_bin_num // 2) else 1,  # cp_policy
        print_info=print_v_table)

    # Frozen Lake:
    fl_env = FrozenLake()
    fl_model = TD0PredictionModel(fl_env, episodes)
    fl_model.perform_td0_policy_evaluation(
        policy=lambda s: fl_env.env.action_space.sample(),  # fl_policy - random policy
        print_info=print_v_table)


# Control (value function estimation \ learning algorithms)

def test_mc_non_exploring_starts_control(episodes=5, print_q_table_and_policy=False):  # episodes=100000
    """
    Performs MC non-exploring starts on multiple environments (separately).
    """
    method_name = 'MC non-exploring starts'

    # Frozen Lake:
    fl_env = FrozenLake()
    fl_model = MCControlModel(fl_env, episodes, eps_max=fl_env.EPS_MIN)
    fl_policy, fl_scores, fl_accumulated_scores = \
        fl_model.perform_mc_non_exploring_starts_control(print_info=print_q_table_and_policy)
    plot_running_average(fl_env.name, method_name, fl_scores, window=episodes//100)
    plot_accumulated_scores(fl_env.name, method_name, fl_accumulated_scores)
    fl_scores, fl_accumulated_scores = run_policy_table(fl_env, fl_policy, episodes)
    plot_running_average(fl_env.name, method_name, fl_scores, window=episodes//100)
    plot_accumulated_scores(fl_env.name, method_name, fl_accumulated_scores)

    # Blackjack:
    bj_env = Blackjack()
    bj_model = MCControlModel(bj_env, episodes, eps_max=0.05, eps_dec=1e-7)
    bj_policy, _, bj_accumulated_scores = \
        bj_model.perform_mc_non_exploring_starts_control(print_info=print_q_table_and_policy)
    plot_accumulated_scores(bj_env.name, method_name, bj_accumulated_scores)
    bj_accumulated_scores = run_policy_table(bj_env, bj_policy, episodes)
    plot_accumulated_scores(bj_env.name, method_name, bj_accumulated_scores)


def test_off_policy_mc_control(episodes=5, print_q_table_and_policy=False):  # episodes=100000
    """
    Performs Off-policy MC Control on multiple environments (separately).
    """
    method_name = 'Off-policy MC Control'

    # Frozen Lake:
    fl_env = FrozenLake()
    fl_model = MCControlModel(fl_env, episodes, eps_max=fl_env.EPS_MIN)
    fl_policy, fl_scores, fl_accumulated_scores = \
        fl_model.perform_off_policy_mc_control(print_info=print_q_table_and_policy)
    plot_running_average(fl_env.name, method_name, fl_scores, window=episodes//100)
    plot_accumulated_scores(fl_env.name, method_name, fl_accumulated_scores)
    fl_scores, fl_accumulated_scores = run_policy_table(fl_env, fl_policy, episodes)
    plot_running_average(fl_env.name, method_name, fl_scores, window=episodes//100)
    plot_accumulated_scores(fl_env.name, method_name, fl_accumulated_scores)

    # Blackjack:
    blackjack_env = Blackjack()
    blackjack_model = MCControlModel(blackjack_env, episodes, eps_max=0.05, eps_dec=1e-7)
    blackjack_policy, _, blackjack_accumulated_scores = \
        blackjack_model.perform_off_policy_mc_control(print_info=print_q_table_and_policy)
    plot_accumulated_scores(blackjack_env.name, method_name, blackjack_accumulated_scores)
    blackjack_accumulated_scores = run_policy_table(blackjack_env, blackjack_policy, episodes)
    plot_accumulated_scores(blackjack_env.name, method_name, blackjack_accumulated_scores)


def test_sarsa(episodes=5):
    """
    Performs SARSA (On-policy TD0 Control) on multiple environments (separately).
    """
    method_name = 'SARSA'

    # Taxi:
    tx_env = Taxi()
    tx_model = TD0ControlModel(tx_env, episodes, alpha=0.4)  # episodes=10000
    tx_q_table, tx_scores, _ = tx_model.perform_sarsa()
    plot_running_average(tx_env.name, method_name, tx_scores)
    tx_scores, _ = run_q_table(tx_env, tx_q_table, episodes)
    plot_running_average(tx_env.name, method_name, tx_scores)

    # Mountain Car:
    mc_env = MountainCar()
    mc_model = TD0ControlModel(mc_env, episodes)
    mc_q_table, mc_scores, _ = mc_model.perform_sarsa()
    plot_running_average(mc_env.name, method_name, mc_scores)
    mc_scores, _ = run_q_table(mc_env, mc_q_table, episodes)
    plot_running_average(mc_env.name, method_name, mc_scores)

    # Cart Pole (Solved):
    cp_env = CartPole()
    cp_model = TD0ControlModel(cp_env, episodes)
    cp_q_table, cp_scores, _ = cp_model.perform_sarsa()
    plot_running_average(cp_env.name, method_name, cp_scores)
    cp_scores, _ = run_q_table(cp_env, cp_q_table, episodes)
    plot_running_average(cp_env.name, method_name, cp_scores)

    # Acrobot:
    ab_env = Acrobot()
    ab_model = TD0ControlModel(ab_env, episodes)
    ab_q_table, ab_scores, _ = ab_model.perform_sarsa()
    plot_running_average(ab_env.name, method_name, ab_scores)
    ab_scores, _ = run_q_table(ab_env, ab_q_table, episodes)
    plot_running_average(ab_env.name, method_name, ab_scores)


def test_expected_sarsa(episodes=5):
    """
    Performs Expected SARSA (On-policy TD0 Control) on multiple environments (separately).
    """
    method_name = 'Expected SARSA'

    # Taxi:
    tx_env = Taxi()
    tx_model = TD0ControlModel(tx_env, episodes, alpha=0.4)  # episodes=10000
    tx_q_table, tx_scores, _ = tx_model.perform_expected_sarsa()
    plot_running_average(tx_env.name, method_name, tx_scores)
    tx_scores, _ = run_q_table(tx_env, tx_q_table, episodes)
    plot_running_average(tx_env.name, method_name, tx_scores)

    # Mountain Car:
    mc_env = MountainCar()
    mc_model = TD0ControlModel(mc_env, episodes)
    mc_q_table, mc_scores, _ = mc_model.perform_expected_sarsa()
    plot_running_average(mc_env.name, method_name, mc_scores)
    mc_scores, _ = run_q_table(mc_env, mc_q_table, episodes)
    plot_running_average(mc_env.name, method_name, mc_scores)

    # Cart Pole (Solved):
    cp_env = CartPole()
    cp_model = TD0ControlModel(cp_env, episodes)
    cp_q_table, cp_scores, _ = cp_model.perform_expected_sarsa()
    plot_running_average(cp_env.name, method_name, cp_scores)
    cp_scores, _ = run_q_table(cp_env, cp_q_table, episodes)
    plot_running_average(cp_env.name, method_name, cp_scores)

    # Acrobot:
    ab_env = Acrobot()
    ab_model = TD0ControlModel(ab_env, episodes)
    ab_q_table, ab_scores, _ = ab_model.perform_expected_sarsa()
    plot_running_average(ab_env.name, method_name, ab_scores)
    ab_scores, _ = run_q_table(ab_env, ab_q_table, episodes)
    plot_running_average(ab_env.name, method_name, ab_scores)


def test_q_learning(episodes=5):
    """
    Performs Q Learning (Off-policy TD0 Control) on multiple environments (separately).
    """
    method_name = 'Q Learning'

    # Taxi:
    tx_env = Taxi()
    tx_model = TD0ControlModel(tx_env, episodes, alpha=0.4)  # episodes=10000
    tx_q_table, tx_scores, _ = tx_model.perform_q_learning()
    plot_running_average(tx_env.name, method_name, tx_scores)
    tx_scores, _ = run_q_table(tx_env, tx_q_table, episodes)
    plot_running_average(tx_env.name, method_name, tx_scores)

    # Mountain Car:
    mc_env = MountainCar()
    mc_model = TD0ControlModel(mc_env, episodes)
    mc_q_table, mc_scores, _ = mc_model.perform_q_learning()
    plot_running_average(mc_env.name, method_name, mc_scores)
    mc_scores, _ = run_q_table(mc_env, mc_q_table, episodes)
    plot_running_average(mc_env.name, method_name, mc_scores)

    # Cart Pole:
    cp_env = CartPole()
    cp_model = TD0ControlModel(cp_env, episodes)
    cp_q_table, cp_scores, _ = cp_model.perform_q_learning()
    plot_running_average(cp_env.name, method_name, cp_scores)
    cp_scores, _ = run_q_table(cp_env, cp_q_table, episodes)
    plot_running_average(cp_env.name, method_name, cp_scores)


def test_double_q_learning(episodes=5):
    """
    Performs Double Q Learning (Off-policy TD0 Control) on multiple environments (separately).
    """
    method_name = 'Double Q Learning'

    # Taxi:
    tx_env = Taxi()
    tx_model = TD0ControlModel(tx_env, episodes, alpha=0.4)  # episodes=10000
    tx_q1_table, tx_q2_table, tx_scores, _ = tx_model.perform_double_q_learning()
    plot_running_average(tx_env.name, method_name, tx_scores)
    tx_q1_scores, _ = run_q_table(tx_env, tx_q1_table, episodes)
    tx_q2_scores, _ = run_q_table(tx_env, tx_q2_table, episodes)
    scores_list = [tx_q1_scores, tx_q2_scores]
    labels = ['Q1', 'Q2']
    plot_running_average_comparison(tx_env.name + ' - ' + method_name, scores_list, labels)

    # Mountain Car:
    mc_env = MountainCar()
    mc_model = TD0ControlModel(mc_env, episodes)
    mc_q1_table, mc_q2_table, mc_scores, _ = mc_model.perform_double_q_learning()
    plot_running_average(mc_env.name, method_name, mc_scores)
    mc_q1_scores, _ = run_q_table(mc_env, mc_q1_table, episodes)
    mc_q2_scores, _ = run_q_table(mc_env, mc_q2_table, episodes)
    scores_list = [mc_q1_scores, mc_q2_scores]
    labels = ['Q1', 'Q2']
    plot_running_average_comparison(mc_env.name + ' - ' + method_name, scores_list, labels)

    # Cart Pole:
    cp_env = CartPole()
    cp_model = TD0ControlModel(cp_env, episodes)
    cp_q1_table, cp_q2_table, cp_scores, _ = cp_model.perform_double_q_learning()
    plot_running_average(cp_env.name, method_name, cp_scores)
    cp_q1_scores, _ = run_q_table(cp_env, cp_q1_table, episodes)
    cp_q2_scores, _ = run_q_table(cp_env, cp_q2_table, episodes)
    scores_list = [cp_q1_scores, cp_q2_scores]
    labels = ['Q1', 'Q2']
    plot_running_average_comparison(cp_env.name + ' - ' + method_name, scores_list, labels)


# Environments

def environment_test(env, episodes=5, eps_max=1.0, eps_dec=None, alpha=0.1,
                     show_scores=False, show_accumulated_scores=False):
    """
    Performs a comparative algorithms test for a single environment.
    """

    labels = [
        'MC non-exploring starts',
        'Off-policy MC',
        'SARSA',
        'Expected SARSA',
        'Q Learning',
        'Double Q Learning'
    ]

    mc_model_01 = MCControlModel(env, episodes, alpha, eps_max=eps_max, eps_dec=eps_dec)
    policy_mc_01, scores_mc_01, accumulated_scores_mc_01 = mc_model_01.perform_mc_non_exploring_starts_control()

    mc_model_02 = MCControlModel(env, episodes, alpha, eps_max=eps_max, eps_dec=eps_dec)
    policy_mc_02, scores_mc_02, accumulated_scores_mc_02 = mc_model_02.perform_off_policy_mc_control()

    sarsa_model = TD0ControlModel(env, episodes, alpha, eps_max=eps_max, eps_dec=eps_dec)
    q_table_sarsa, scores_sarsa, accumulated_scores_sarsa = sarsa_model.perform_sarsa()

    e_sarsa_model = TD0ControlModel(env, episodes, alpha, eps_max=eps_max, eps_dec=eps_dec)
    q_table_e_sarsa, scores_e_sarsa, accumulated_scores_e_sarsa = e_sarsa_model.perform_expected_sarsa()

    q_l_model = TD0ControlModel(env, episodes, alpha, eps_max=eps_max, eps_dec=eps_dec)
    q_table_q_l, scores_q_l, accumulated_scores_q_l = q_l_model.perform_q_learning()

    d_q_l_model = TD0ControlModel(env, episodes, alpha, eps_max=eps_max, eps_dec=eps_dec)
    q1_table_d_q_l, q2_table_d_q_l, scores_d_q_l, accumulated_scores_d_q_l = d_q_l_model.perform_double_q_learning()
    q_table_d_q_l = {}
    for s in q1_table_d_q_l:
        q_table_d_q_l[s] = (q1_table_d_q_l[s] + q2_table_d_q_l[s]) / 2

    if show_scores:
        scores_list = [scores_mc_01, scores_mc_02, scores_sarsa, scores_e_sarsa, scores_q_l, scores_d_q_l]
        plot_running_average_comparison(env.name + ' - Training', scores_list, labels,  # window=episodes//100,
                                        file_name=env.file_name + '-score-training')

    if show_accumulated_scores:
        accumulated_scores_list = [accumulated_scores_mc_01, accumulated_scores_mc_02,
                                   accumulated_scores_sarsa, accumulated_scores_e_sarsa,
                                   accumulated_scores_q_l, accumulated_scores_d_q_l]
        plot_accumulated_scores_comparison(env.name + ' - Training', accumulated_scores_list, labels,
                                           file_name=env.file_name + '-accumulated-score-training')

    scores_mc_01, accumulated_scores_mc_01 = run_policy_table(env, policy_mc_01, episodes)
    scores_mc_02, accumulated_scores_mc_02 = run_policy_table(env, policy_mc_02, episodes)
    scores_sarsa, accumulated_scores_sarsa = run_q_table(env, q_table_sarsa, episodes)
    scores_e_sarsa, accumulated_scores_e_sarsa = run_q_table(env, q_table_e_sarsa, episodes)
    scores_q_l, accumulated_scores_q_l = run_q_table(env, q_table_q_l, episodes)
    scores_d_q_l, accumulated_scores_d_q_l = run_q_table(env, q_table_d_q_l, episodes)

    if show_scores:
        scores_list = [scores_mc_01, scores_mc_02, scores_sarsa, scores_e_sarsa, scores_q_l, scores_d_q_l]
        plot_running_average_comparison(env.name + ' - Test', scores_list, labels,  # window=episodes//100,
                                        file_name=env.file_name + '-scores-test')

    if show_accumulated_scores:
        accumulated_scores_list = [accumulated_scores_mc_01, accumulated_scores_mc_02,
                                   accumulated_scores_sarsa, accumulated_scores_e_sarsa,
                                   accumulated_scores_q_l, accumulated_scores_d_q_l]
        plot_accumulated_scores_comparison(env.name + ' - Test', accumulated_scores_list, labels,
                                           file_name=env.file_name + '-accumulated-scores-test')


def test_environments():
    """
    Performs environment_test() on multiple environments (separately).
    """
    environment_test(FrozenLake(), eps_max=0.1, eps_dec=None)  # episodes=100000
    environment_test(Taxi(), alpha=0.4)  # episodes=10000
    environment_test(Blackjack(), eps_max=0.05, eps_dec=1e-7)  # episodes=100000
    environment_test(MountainCar())  # episodes=50000
    environment_test(CartPole())  # episodes=50000
    environment_test(Acrobot())  # episodes=50000
