import datetime
import tensorflow as tf

from reinforcement_learning.utils.plotter import plot_running_average, plot_running_average_comparison
from reinforcement_learning.utils.utils import pickle_load, pickle_save


def load_training_data(agent, load_checkpoint):
    scores_history = []
    learn_episode_index = -1
    max_avg = None
    if load_checkpoint:
        try:
            print('...Loading models...')  # REMOVE when load logic is finalized
            agent.load_models()  # REMOVE when load logic is finalized
            print('...Loading learn_episode_index...')
            learn_episode_index = pickle_load('learn_episode_index', agent.chkpt_dir)
            print('...Loading scores_history...')
            scores_history = pickle_load('scores_history_train', agent.chkpt_dir)
            print('...Loading max_avg...')
            max_avg = pickle_load('max_avg', agent.chkpt_dir)
        except (ValueError, tf.OpError, OSError):  # REMOVE when load logic is finalized
            print('...No models to load...')  # REMOVE when load logic is finalized
        except FileNotFoundError:
            print('...No data to load...')
    return scores_history, learn_episode_index, max_avg


def load_scores_history_and_plot(env_name, method_name, window, chkpt_dir,
                                 training_data=False, show_scores=False):
    try:
        print('...Loading scores_history...')
        suffix = '_train_total' if training_data else '_test'
        scores_history = pickle_load('scores_history' + suffix, chkpt_dir)
        plot_running_average(env_name, method_name, scores_history, window, show=show_scores,
                             file_name='scores_history' + suffix, directory=chkpt_dir)

    except FileNotFoundError:
        print('...No scores history data to load...')


def load_multiple_scores_history_and_plot(custom_env, method_name, base_dir, sub_dirs, labels,
                                          training_data, show_scores):

    suffix = '_train_total' if training_data else '_test'

    try:
        print('...Loading scores_history...')

        scores_list = []
        for sub_dir in sub_dirs:
            scores_list.append(pickle_load('scores_history' + suffix, base_dir + sub_dir))

        plot_running_average_comparison(custom_env.name + ' - ' + method_name, scores_list, labels,
                                        show=show_scores,
                                        file_name='scores_history' + suffix, directory=base_dir)
    except FileNotFoundError:
        print('...No scores history data to load...')


def save_training_data(agent, learn_episode_index, scores_history):
    save_start_time = datetime.datetime.now()
    pickle_save(learn_episode_index, 'learn_episode_index', agent.chkpt_dir)
    pickle_save(scores_history, 'scores_history_train', agent.chkpt_dir)
    agent.save_models()
    print('Save time: %s' % str(datetime.datetime.now() - save_start_time).split('.')[0])


def extract_data(base_dir, sub_dir):
    chkpt_dir = base_dir + sub_dir

    learn_episode_index = pickle_load('learn_episode_index', chkpt_dir)
    max_avg = pickle_load('max_avg', chkpt_dir)

    print('learn_episode_index', learn_episode_index, 'max_avg', max_avg)
