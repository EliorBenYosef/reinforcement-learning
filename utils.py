import os

from IPython.display import clear_output
import time
import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras.backend.tensorflow_backend as keras_tensorflow_backend
from keras.backend import set_session as keras_set_session
import keras.optimizers as optimizers

import torch as T
import torch.optim.rmsprop as T_optim_rmsprop
import torch.optim.adagrad as T_optim_adagrad
import torch.optim.adadelta as T_optim_adadelta


LIBRARY_TF = 0
LIBRARY_KERAS = 1
LIBRARY_TORCH = 2


class Plotter:

    # colors = ['r--', 'g--', 'b--', 'c--', 'm--', 'y--', 'k--', 'w--']

    # colors = ['#FF0000', '#fa3c3c', '#E53729',
    #           '#f08228', '#FB9946', '#FF7F00',
    #           '#e6af2d',
    #           '#e6dc32', '#FFFF00',
    #           '#a0e632', '#00FF00',  '#00dc00',
    #           '#17A858', '#00d28c',
    #           '#00c8c8', '#0DB0DD',  '#00a0ff', '#1e3cff', '#0000FF',
    #           '#6e00dc', '#8B00FF',  '#4B0082', '#a000c8', '#662371',
    #           '#f00082']

    colors = ['#FF0000', '#E53729',
              '#f08228', '#FF7F00',
              '#e6af2d',
              '#e6dc32', '#FFFF00',
              '#a0e632', '#00dc00',
              '#17A858', '#00d28c',
              '#00c8c8', '#1e3cff',
              '#6e00dc', '#a000c8',
              '#f00082']

    @staticmethod
    def get_running_avg(scores, window):
        episodes = len(scores)

        x = [i + 1 for i in range(episodes)]

        running_avg = np.empty(episodes)
        for t in range(episodes):
            running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])

        return x, running_avg

    @staticmethod
    def plot_running_average(env_name, method_name, scores, window=100, show=False, file_name=None, directory=''):
        plt.title(env_name + ' - ' + method_name + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.plot(*Plotter.get_running_avg(scores, window))
        if file_name:
            plt.savefig(directory + file_name + '.png')
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_accumulated_scores(env_name, method_name, scores, show=False, file_name=None, directory=''):
        plt.title(env_name + ' - ' + method_name + ' - Accumulated Score')
        plt.ylabel('Accumulated Score')
        plt.xlabel('Episode')
        x = [i + 1 for i in range(len(scores))]
        plt.plot(x, scores)
        if file_name:
            plt.savefig(directory + file_name + '.png')
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_running_average_comparison(main_title, scores_list, labels=None, window=100, show=False,
                                        file_name=None, directory=''):
        plt.figure(figsize=(8.5, 4.5))
        plt.title(main_title + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
        plt.ylabel('Score')
        plt.xlabel('Episode')
        # colors = []
        # for i in range(len(scores_list)):
        #     colors.append(np.random.rand(3, ))
        for i, scores in enumerate(scores_list):
            plt.plot(*Plotter.get_running_avg(scores, window), Plotter.colors[i])
        if labels:
            # plt.legend(labels)
            plt.legend(labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            plt.subplots_adjust(right=0.7)
        if file_name:
            plt.savefig(directory + file_name + '.png')
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_accumulated_scores_comparison(main_title, scores_list, labels=None, show=False,
                                           file_name=None, directory=''):
        plt.figure(figsize=(8.5, 4.5))
        plt.title(main_title + ' - Accumulated Score')
        plt.ylabel('Accumulated Score')
        plt.xlabel('Episode')
        # colors = []
        # for i in range(len(scores_list)):
        #     colors.append(np.random.rand(3, ))
        for i, scores in enumerate(scores_list):
            x = [i + 1 for i in range(len(scores))]
            plt.plot(x, scores, Plotter.colors[i])
        if labels:
            # plt.legend(labels)
            plt.legend(labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            plt.subplots_adjust(right=0.7)
        if file_name:
            plt.savefig(directory + file_name + '.png')
        if show:
            plt.show()
        plt.close()


class Printer:

    @staticmethod
    def print_average_score(total_scores, ratio=10):
        # Calculate and print the average score per a number of episodes (tick)

        scores_per_tick_episodes = np.split(np.array(total_scores), ratio)  # episodes / tick

        episodes = len(total_scores)
        tick = episodes // ratio
        print('\n********Average score per %d episodes********\n' % tick)
        count = tick
        for r in scores_per_tick_episodes:
            print(count, ": ", str(sum(r / 1000)))
            count += tick

    @staticmethod
    def print_training_progress(i, ep_score, scores_history, avg_num, trailing=True, eps=None):
        print('episode: %d ;' % (i + 1), 'score: %d' % ep_score)  # score: %.2f

        eps_string = ''
        if eps:
            eps_string = 'epsilon %.3f' % eps  # %.4f

        if trailing and (i + 1) >= avg_num:
            # gives you the running avg of the last 'avg_num' episodes, every episode:
            avg_score = np.mean(scores_history[-avg_num:])
            print('trailing %d episodes ;' % avg_num,
                  'average score %.3f ;' % avg_score,
                  eps_string)

        elif (i + 1) % avg_num == 0:
            # gives you the running avg of the last 'avg_num' episodes, every 'avg_num' episodes:
            avg_score = np.mean(scores_history[max(0, i + 1 - avg_num):(i + 1)])
            print('episodes: %d - %d ;' % (i + 2 - avg_num, i + 1),
                  'average score %.3f ;' % avg_score,
                  eps_string)

    @staticmethod
    def print_v(V):
        print('\n', 'V table', '\n')
        for s in V:
            print(s, '%.5f' % V[s])

    @staticmethod
    def print_q(Q):
        print('\n', 'Q table', '\n')
        # print(Q)
        for s, a in Q:
            print('s', s, 'a', a, ' - ', '%.3f' % Q[s, a])

    @staticmethod
    def print_policy(Q, policy):
        print('\n', 'Policy', '\n')
        for s in policy:
            a = policy[s]
            print('s', s, 'a', a, ' - ', '%.3f' % Q[s, a])


class DeviceGetUtils:

    @staticmethod
    def tf_get_local_devices(GPUs_only=False):
        # # confirm TensorFlow sees the GPU
        # assert 'GPU' in str(device_lib.list_local_devices())

        local_devices = device_lib.list_local_devices()  # local_device_protos
        # possible properties: name, device_type, memory_limit, incarnation, locality, physical_device_desc.
        #   name - str with the following structure: '/' + prefix ':' + device_type + ':' + device_type_order_num.
        #   device_type - CPU \ XLA_CPU \ GPU \ XLA_GPU
        #   locality (can be empty) - for example: { bus_id: 1 links {} }
        #   physical_device_desc (optional) - for example:
        #       "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
        if GPUs_only:
            return [dev.name for dev in local_devices if 'GPU' in dev.device_type]
        else:
            return [dev.name for dev in local_devices]

    @staticmethod
    def keras_get_available_GPUs():  # To Check if keras(>=2.1.1) is using GPU:
        # # confirm Keras sees the GPU
        # assert len(keras_tensorflow_backend._get_available_gpus()) > 0

        return keras_tensorflow_backend._get_available_gpus()

    @staticmethod
    def torch_get_current_device_name():  # To Check if keras(>=2.1.1) is using GPU:
        # confirm PyTorch sees the GPU
        if T.cuda.is_available() and T.cuda.device_count() > 0:
            return T.cuda.get_device_name(T.cuda.current_device())


class DeviceSetUtils:

    @staticmethod
    def set_device(lib_type, devices_dict=None):  # {type: bus_id}
        # it seems that for:
        #   TF - tf_get_session_according_to_device() alone is enough...
        #   Keras - tf_set_device() alone is enough...
        # maybe only one method is enough for either?

        if devices_dict is None:
            devices_dict = {'GPU': 0}
            # devices = {'GPU': 0, 'GPU': 1}  # 'CPU': 0

        designated_GPUs_bus_id_str = ''
        for device_type, device_bus_id in devices_dict.items():
            if len(designated_GPUs_bus_id_str) > 0:
                designated_GPUs_bus_id_str += ','
            designated_GPUs_bus_id_str += str(device_bus_id)

        if lib_type == LIBRARY_TF:
            DeviceSetUtils.tf_set_device(designated_GPUs_bus_id_str)
        elif lib_type == LIBRARY_KERAS:
            DeviceSetUtils.tf_set_device(designated_GPUs_bus_id_str)
            DeviceSetUtils.keras_set_session_according_to_device(devices_dict)

    # when trying to run a tensorflow \ keras model on GPU, make sure:
    #   1. the system has a Nvidia GPU (AMD doesn't work yet).
    #   2. the GPU version of tensorflow is installed.
    #   3. CUDA is installed. https://www.tensorflow.org/install
    #   4. tensorflow is running with GPU. use tf_get_local_devices()

    @staticmethod
    def tf_set_device(designated_GPUs_bus_id_str):  # can be singular: '0', or multiple: '0,1'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # set GPUs (CUDA devices) IDs' order by pci bus IDs (so it's consistent with nvidia-smi's output).
        os.environ['CUDA_VISIBLE_DEVICES'] = designated_GPUs_bus_id_str  # specify which GPU ID(s) to be used.
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    @staticmethod
    def tf_get_session_according_to_device(devices_dict):
        if devices_dict is not None:
            gpu_options = tf.GPUOptions(allow_growth=True)  # starts with allocating an approximated amount of GPU memory, and expands if necessary
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)  # set the fraction of GPU memory to be allocated
            config = tf.ConfigProto(device_count=devices_dict, gpu_options=gpu_options, log_device_placement=False)  # log device placement tells which device is used.
            # config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.5
            sess = tf.Session(config=config)
        else:
            sess = tf.Session()
        return sess

    @staticmethod
    def keras_set_session_according_to_device(device_map):
        # call this function after importing keras if you are working on a machine.
        keras_set_session(DeviceSetUtils.tf_get_session_according_to_device(device_map))

    @staticmethod
    def torch_get_device_according_to_device_type(device_str):  # e.g.: 'cpu' \ 'gpu' \ 'cuda:1'
        # enabling GPU vs CPU:
        if device_str == 'cpu':
            device = T.device('cpu')  # default CPU. cpu:0 ?
        elif device_str == 'cuda:1':
            device = T.device('cuda:1' if T.cuda.is_available() else 'cuda')  # 2nd\default GPU. cuda:0 ?
        else:
            device = T.device('cuda' if T.cuda.is_available() and T.cuda.device_count() > 0 else 'cpu')  # default GPU \ default CPU. :0 ?
        return device


class Optimizers:

    OPTIMIZER_Adam = 0
    OPTIMIZER_RMSprop = 1
    OPTIMIZER_Adadelta = 2
    OPTIMIZER_Adagrad = 3
    OPTIMIZER_SGD = 4

    @staticmethod
    def tf_get_optimizer(optimizer_type, lr, momentum=None):  # momentum=0.9
        if optimizer_type == Optimizers.OPTIMIZER_SGD:
            if momentum is None:
                return tf.train.GradientDescentOptimizer(lr)
            else:
                return tf.train.MomentumOptimizer(lr, momentum)
        elif optimizer_type == Optimizers.OPTIMIZER_Adagrad:
            return tf.train.AdagradOptimizer(lr)
        elif optimizer_type == Optimizers.OPTIMIZER_Adadelta:
            return tf.train.AdadeltaOptimizer(lr)
        elif optimizer_type == Optimizers.OPTIMIZER_RMSprop:
            if momentum is None:
                return tf.train.RMSPropOptimizer(lr)
            else:
                return tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=momentum, epsilon=1e-6)
        else:  # optimizer_type == Optimizers.OPTIMIZER_Adam
            return tf.train.AdamOptimizer(lr)

    @staticmethod
    def keras_get_optimizer(optimizer_type, lr, momentum=0.9):
        if optimizer_type == Optimizers.OPTIMIZER_SGD:
            return optimizers.SGD(lr, momentum)
        elif optimizer_type == Optimizers.OPTIMIZER_Adagrad:
            return optimizers.Adagrad(lr)
        elif optimizer_type == Optimizers.OPTIMIZER_Adadelta:
            return optimizers.Adadelta(lr)
        elif optimizer_type == Optimizers.OPTIMIZER_RMSprop:
            return optimizers.RMSprop(lr)
            # return optimizers.RMSprop(lr=lr, decay=0.99, momentum=0.0, epsilon=1e-6)
        else:  # optimizer_type == Optimizers.OPTIMIZER_Adam
            return optimizers.Adam(lr)

    @staticmethod
    def torch_get_optimizer(optimizer_type, params, lr, momentum=None):  # momentum=0.9
        if optimizer_type == Optimizers.OPTIMIZER_SGD:
            if momentum is None:
                return T.optim.SGD(params, lr)
            else:
                return T.optim.SGD(params, lr, momentum)
        elif optimizer_type == Optimizers.OPTIMIZER_Adagrad:
            return T_optim_adagrad.Adagrad(params, lr)
        elif optimizer_type == Optimizers.OPTIMIZER_Adadelta:
            return T_optim_adadelta.Adadelta(params, lr)
        elif optimizer_type == Optimizers.OPTIMIZER_RMSprop:
            if momentum is None:
                return T_optim_rmsprop.RMSprop(params, lr)
            else:
                return T_optim_rmsprop.RMSprop(params, lr, weight_decay=0.99, momentum=momentum, eps=1e-6)
        else:  # optimizer_type == Optimizers.OPTIMIZER_Adam
            return T.optim.Adam(params, lr)


class Calculator:

    EPS_DEC_LINEAR = 0
    EPS_DEC_EXPONENTIAL = 1
    EPS_DEC_EXPONENTIAL_TIME_RELATED = 2
    # EPS_DEC_QUADRATIC = 4

    @staticmethod
    def decrement_eps(eps_current, eps_min, eps_dec, eps_dec_type, eps_max=None, t=None):
        if eps_dec_type == Calculator.EPS_DEC_EXPONENTIAL:
            eps_temp = eps_current * eps_dec  # eps_dec = 0.996
        elif eps_dec_type == Calculator.EPS_DEC_EXPONENTIAL_TIME_RELATED and eps_max is not None and t is not None:
            return eps_min + (eps_max - eps_min) * np.exp(-eps_dec * t)  # t == i
        else:  # eps_dec_type == Calculator.EPS_DEC_LINEAR:
            eps_temp = eps_current - eps_dec

        return max(eps_temp, eps_min)

    @staticmethod
    def calc_conv_layer_output_dim(Dimension, Filter, Padding, Stride):
        return (Dimension - Filter + 2 * Padding) / Stride + 1

    @staticmethod
    def calc_conv_layer_output_dims(Height, Width, Filter, Padding, Stride):
        h = (Height - Filter + 2 * Padding) // Stride + 1
        w = (Width - Filter + 2 * Padding) // Stride + 1
        return h, w

    @staticmethod
    def calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA):
        memory_G = []
        G = 0
        n = len(memory_r)
        for reward in reversed(memory_r):
            if memory_terminal[n - 1 - len(memory_G)]:
                G = 0
            G = GAMMA * G + reward
            memory_G.append(G)
        memory_G = np.flip(np.array(memory_G, dtype=np.float64), 0)
        memory_G = General.scale_and_normalize(memory_G)
        return memory_G


class ActionChooser:

    @staticmethod
    def get_max_action_from_q_table(Q, s, action_space_size):
        values = np.array([Q[s, a] for a in range(action_space_size)])
        # values == Q[s, :]                                             # if Q is a numpy.ndarray
        a_max = np.random.choice(np.where(values == values.max())[0])
        return a_max


class Tester:

    @staticmethod
    def test_method(custom_env, episodes, choose_action):
        env = custom_env.env
        print('\n', 'Test Started', '\n')
        start_time = datetime.datetime.now()
        total_scores = np.zeros(episodes)
        total_accumulated_scores = np.zeros(episodes)
        accumulated_score = 0
        eval = custom_env.get_evaluation_tuple()
        for i in range(episodes):
            done = False
            ep_steps = 0
            ep_score = 0
            observation = env.reset()
            s = custom_env.get_state(observation)
            while not done:
                a = choose_action(s)
                observation_, reward, done, info = env.step(a)
                eval = custom_env.update_evaluation_tuple(i + 1, reward, done, eval)
                ep_steps += 1
                ep_score += reward
                accumulated_score += reward
                s_ = custom_env.get_state(observation_)
                observation, s = observation_, s_
            total_scores[i] = ep_score
            total_accumulated_scores[i] = accumulated_score
            Printer.print_training_progress(i, ep_score, total_scores, custom_env.window)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print('\n', 'Test Ended ~~~ Episodes: %d ~~~ Duration: %s' % (episodes, duration), '\n')
        custom_env.analyze_evaluation_tuple(eval, episodes)
        return total_scores, total_accumulated_scores

    @staticmethod
    def test_q_table(custom_env, Q, episodes=1000):
        return Tester.test_method(
            custom_env, episodes,
            lambda s: ActionChooser.get_max_action_from_q_table(Q, s, custom_env.env.action_space.n)
        )

    @staticmethod
    def test_policy(custom_env, policy, episodes=1000):
        return Tester.test_method(custom_env, episodes,
                                  lambda s: policy[s])

    @staticmethod
    def test_trained_agent(custom_env, agent, enable_models_saving, episodes=1000):
        total_scores, total_accumulated_scores = Tester.test_method(
            custom_env, episodes,
            lambda s: agent.choose_action(s))
        if enable_models_saving:
            SaverLoader.pickle_save(total_scores, 'scores_history_test', agent.chkpt_dir)
        return total_scores, total_accumulated_scores


class Watcher:

    @staticmethod
    def watch_method(custom_env, episodes, choose_action, is_toy_text=False):
        env = custom_env.env

        for i in range(episodes):
            done = False
            ep_steps = 0
            ep_score = 0
            observation = env.reset()
            s = custom_env.get_state(observation)

            if is_toy_text:
                print('\n*****EPISODE ', i + 1, '*****\n')
                time.sleep(1)
                clear_output(wait=True)
            env.render()
            if is_toy_text:
                time.sleep(0.3)

            while not done:
                a = choose_action(s)
                observation_, reward, done, info = env.step(a)
                ep_steps += 1
                ep_score += reward
                s_ = custom_env.get_state(observation_)
                observation, s = observation_, s_

                if is_toy_text:
                    clear_output(wait=True)
                env.render()
                if is_toy_text:
                    time.sleep(0.3)

            print('Episode Score:', ep_score)
            if is_toy_text:
                time.sleep(3)
                clear_output(wait=True)

        env.close()

    @staticmethod
    def watch_q_table(custom_env, Q, episodes=3):
        Watcher.watch_method(
            custom_env, episodes,
            lambda s: ActionChooser.get_max_action_from_q_table(Q, s, custom_env.env.action_space.n)
        )

    @staticmethod
    def watch_policy(custom_env, policy, episodes=3):
        Watcher.watch_method(custom_env, episodes,
                             lambda s: policy[s])

    @staticmethod
    def watch_trained_agent(custom_env, agent, episodes=3):
        Watcher.watch_method(custom_env, episodes,
                             lambda s: agent.choose_action(s))


class SaverLoader:

    @staticmethod
    def pickle_load(file_name, directory=''):
        with open(directory + file_name + '.pkl', 'rb') as file:  # .pickle  # rb = read binary
            var = pickle.load(file)  # var == [X_train, y_train]
        return var

    @staticmethod
    def pickle_save(var, file_name, directory=''):
        with open(directory + file_name + '.pkl', 'wb') as file:  # .pickle  # wb = write binary
            pickle.dump(var, file)  # var == [X_train, y_train]


class General:

    @staticmethod
    def scale_and_normalize(np_array):
        mean = np.mean(np_array)
        std = np.std(np_array)
        if std == 0:
            std = 1
        return (np_array - mean) / std

    @staticmethod
    def compare_current_and_original_params(current_actor, current_critic,
                                            original_actor, original_critic):
        current_actor_dict = dict(current_actor.named_parameters())
        original_actor_dict = dict(original_actor.named_parameters())
        print('Checking Actor parameters')
        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))

        current_critic_dict = dict(current_critic.named_parameters())
        original_critic_dict = dict(original_critic.named_parameters())
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))

        input()

    @staticmethod
    def get_policy_from_q_table(states, Q, action_space_size):
        policy = {}
        for s in states:
            policy[s] = ActionChooser.get_max_action_from_q_table(Q, s, action_space_size)

        return policy

    @staticmethod
    def query_env(env):

        print(
            'Environment Id -', env.spec.id, '\n',  # id (str): The official environment ID
            'Non-Deterministic -', env.spec.nondeterministic, '\n',  # nondeterministic (bool): Whether this environment is non-deterministic even after seeding
            'Observation Space -', env.observation_space, '\n',
            'Action Space -', env.action_space, '\n',

            'Max Episode Seconds -', env.spec.max_episode_seconds, '\n',
            'Max Episode Steps -', env.spec.max_episode_steps, '\n',  # max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of

            'Reward Range -', env.reward_range, '\n',
            'Reward Threshold -', env.spec.reward_threshold, '\n',  # reward_threshold (Optional[int]): The reward threshold before the task is considered solved

            'TimeStep Limit -', env.spec.timestep_limit, '\n',
            'Trials -', env.spec.trials, '\n',

            'Local Only -', getattr(env.spec, '_local_only', 'not defined'), '\n',
            'kwargs -', getattr(env.spec, '_kwargs', '')  # kwargs (dict): The kwargs to pass to the environment class
        )

    @staticmethod
    def get_file_name(env_file_name, agent, episodes, method_name):
        # options:
        #   .replace('.', 'p')
        #   .split('.')[1]

        if env_file_name is not None:
            env = env_file_name + '_'
        else:
            env = ''

        ############################

        gamma = 'G' + str(agent.GAMMA).replace('.', 'p') + '_'  # 'GAMMA-'

        fc_layers_dims = 'FC-'
        for i, fc_layer_dims in enumerate(agent.fc_layers_dims):
            if i:
                fc_layers_dims += 'x'
            fc_layers_dims += str(fc_layer_dims)
        fc_layers_dims += '_'

        ############################

        optimizer = 'OPT_'
        if agent.optimizer_type == Optimizers.OPTIMIZER_Adam:
            optimizer += 'adam_'
        elif agent.optimizer_type == Optimizers.OPTIMIZER_RMSprop:
            optimizer += 'rms_'  # 'rmsprop_'
        elif agent.optimizer_type == Optimizers.OPTIMIZER_Adadelta:
            optimizer += 'adad_'  # 'adadelta_'
        elif agent.optimizer_type == Optimizers.OPTIMIZER_Adagrad:
            optimizer += 'adag_'  # 'adagrad_'
        else:  # agent.optimizer_type == Optimizers.OPTIMIZER_SGD
            optimizer += 'sgd_'
        alpha = 'a-' + str(agent.ALPHA).replace('.', 'p') + '_'  # 'alpha-'

        if method_name == 'AC' or method_name == 'DDPG':
            beta = 'b-' + str(agent.BETA).replace('.', 'p') + '_'  # 'beta-'
        else:
            beta = ''

        ############################

        if method_name == 'DQL':
            eps_max = 'max-' + str(agent.eps_max).replace('.', 'p') + '_'
            eps_min = 'min-' + str(agent.eps_min).replace('.', 'p') + '_'
            eps_dec = 'dec-' + str(agent.eps_dec).replace('.', 'p') + '_'
            eps = 'EPS_' + eps_max + eps_min + eps_dec
        else:
            eps = ''

        ############################

        if method_name == 'DQL' or method_name == 'DDPG':
            memory_size = 'size-' + str(agent.memory_size)
            memory_batch_size = 'batch-' + str(agent.memory_batch_size)
            replay_buffer = 'MEM_' + memory_size + memory_batch_size
        else:
            replay_buffer = ''

        if method_name == 'PG':
            ep_batch_num = 'PG-ep-batch-' + str(agent.ep_batch_num) + '_'
        else:
            ep_batch_num = ''

        ############################

        episodes = 'N-' + str(episodes)  # n_episodes

        file_name = env + gamma + fc_layers_dims + \
                         optimizer + alpha + beta + \
                         eps + replay_buffer + ep_batch_num + episodes

        return file_name

