import datetime
import os
import numpy as np
from gym import wrappers

import tensorflow as tf
import keras.models as keras_models
import keras.layers as keras_layers
import keras.initializers as keras_init
import keras.backend as keras_backend
import torch
import torch.nn.functional as torch_func
import torch.nn.init as torch_init

from reinforcement_learning.utils.utils import print_training_progress, pickle_save, make_sure_dir_exists,\
    decrement_eps, EPS_DEC_LINEAR
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH,\
    OPTIMIZER_Adam, INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES, ATARI_FRAMES_STACK_SIZE
from reinforcement_learning.deep_RL.utils.utils import calc_conv_layer_output_dims, eps_greedy
from reinforcement_learning.deep_RL.utils.saver_loader import load_training_data, save_training_data
from reinforcement_learning.deep_RL.utils.optimizers import tf_get_optimizer, keras_get_optimizer, torch_get_optimizer
from reinforcement_learning.deep_RL.utils.devices import tf_get_session_according_to_device, \
    torch_get_device_according_to_device_type
from reinforcement_learning.deep_RL.utils.replay_buffer import ReplayBuffer


class NN(object):
    """
    Q NN \ Deep Q-Network (DQN)
    """

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, alpha, chkpt_dir):
        self.input_type = custom_env.input_type

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.n_actions = custom_env.n_actions

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.chkpt_dir = chkpt_dir

    def create_nn_tensorflow(self, name):
        return NN.NN_TensorFlow(self, name)

    def create_nn_keras(self):
        return NN.NN_Keras(self)

    def create_nn_torch(self, relevant_screen_size=None, image_channels=None):
        return NN.NN_Torch(self, relevant_screen_size, image_channels)

    class NN_TensorFlow(object):

        def __init__(self, nn, name, device_map=None):
            self.nn = nn

            self.name = name

            self.sess = tf_get_session_according_to_device(device_map)
            self.build_network()
            self.sess.run(tf.compat.v1.global_variables_initializer())

            self.saver = tf.compat.v1.train.Saver()
            self.checkpoint_file = os.path.join(nn.chkpt_dir, 'q_nn_tf.ckpt')

            self.params = tf.trainable_variables(scope=self.name)

        def build_network(self):
            with tf.compat.v1.variable_scope(self.name):
                self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.nn.input_dims], name='s')
                self.a_indices_one_hot = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nn.n_actions],
                                                                  name='a_indices_one_hot')
                self.q_target_chosen_a = tf.compat.v1.placeholder(tf.float32, shape=[None], name='q_target_chosen_a')

                if self.nn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                    x = tf.layers.dense(self.s, units=self.nn.fc_layers_dims[0], activation='relu',
                                        kernel_initializer=tf.initializers.he_normal())

                else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                    x = tf.layers.conv2d(self.s, filters=32, kernel_size=(8, 8), strides=4, name='conv1',
                                         kernel_initializer=tf.initializers.he_normal())
                    x = tf.layers.batch_normalization(x, epsilon=1e-5, name='conv1_bn')
                    x = tf.nn.relu(x)

                    x = tf.layers.conv2d(x, filters=64, kernel_size=(4, 4), strides=2, name='conv2',
                                         kernel_initializer=tf.initializers.he_normal())
                    x = tf.layers.batch_normalization(x, epsilon=1e-5, name='conv2_bn')
                    x = tf.nn.relu(x)

                    x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), strides=1, name='conv3',
                                         kernel_initializer=tf.initializers.he_normal())
                    x = tf.layers.batch_normalization(x, epsilon=1e-5, name='conv3_bn')
                    x = tf.nn.relu(x)

                    x = tf.layers.flatten(x)

                x = tf.layers.dense(x, units=self.nn.fc_layers_dims[-1], activation='relu',
                                    kernel_initializer=tf.initializers.he_normal())

                self.q_values = tf.layers.dense(x, units=self.nn.n_actions,
                                                kernel_initializer=tf.initializers.glorot_normal())

                q_chosen_a = tf.reduce_sum(tf.multiply(self.q_values, self.a_indices_one_hot))
                loss = tf.reduce_mean(tf.square(q_chosen_a - self.q_target_chosen_a))  # MSE

                optimizer = tf_get_optimizer(self.nn.optimizer_type, self.nn.ALPHA)
                self.optimize = optimizer.minimize(loss)  # train_op

        def forward(self, batch_s):
            q_eval_s = self.sess.run(self.q_values, feed_dict={self.s: batch_s})
            return q_eval_s

        def learn_batch(self, batch_s, batch_a_indices_one_hot, q_target_chosen_a):
            # print('Training Started')
            self.sess.run(self.optimize,
                          feed_dict={self.s: batch_s,
                                     self.a_indices_one_hot: batch_a_indices_one_hot,
                                     self.q_target_chosen_a: q_target_chosen_a})
            # print('Training Finished')

        def load_model_file(self):
            print("...Loading TF checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving TF checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class NN_Keras(object):

        def __init__(self, nn):
            self.nn = nn

            self.h5_file = os.path.join(nn.chkpt_dir, 'q_nn_keras.h5')

            self.model, self.q_values_model = self.build_network()

        def build_network(self):
            s = keras_layers.Input(shape=self.nn.input_dims, dtype='float32', name='s')

            if self.nn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                x = keras_layers.Dense(self.nn.fc_layers_dims[0], activation='relu',
                                       kernel_initializer=keras_init.he_normal())(s)

            else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                x = keras_layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, name='conv1',
                                        kernel_initializer=keras_init.he_normal())(s)
                x = keras_layers.BatchNormalization(epsilon=1e-5, name='conv1_bn')(x)
                x = keras_layers.Activation('relu', name='conv1_bn_ac')(x)

                x = keras_layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, name='conv2',
                                        kernel_initializer=keras_init.he_normal())(x)
                x = keras_layers.BatchNormalization(epsilon=1e-5, name='conv2_bn')(x)
                x = keras_layers.Activation('relu', name='conv2_bn_ac')(x)

                x = keras_layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, name='conv3',
                                        kernel_initializer=keras_init.he_normal())(x)
                x = keras_layers.BatchNormalization(epsilon=1e-5, name='conv3_bn')(x)
                x = keras_layers.Activation('relu', name='conv3_bn_ac')(x)

                x = keras_layers.Flatten()(x)

            x = keras_layers.Dense(self.nn.fc_layers_dims[-1], activation='relu',
                                   kernel_initializer=keras_init.he_normal())(x)

            q_values = keras_layers.Dense(self.nn.n_actions, name='q_values',
                                          kernel_initializer=keras_init.glorot_normal())(x)

            q_values_model = keras_models.Model(inputs=s, outputs=q_values)

            #############################

            a_indices_one_hot = keras_layers.Input(
                shape=(self.nn.n_actions,), dtype='float32', name='a_indices_one_hot')

            x = keras_layers.Multiply()([q_values, a_indices_one_hot])
            q_chosen_a = keras_layers.Lambda(lambda z: keras_backend.sum(z), output_shape=(1,))(x)

            model = keras_models.Model(inputs=[s, a_indices_one_hot], outputs=q_chosen_a)
            optimizer = keras_get_optimizer(self.nn.optimizer_type, self.nn.ALPHA)
            model.compile(optimizer=optimizer, loss='mse')

            return model, q_values_model

        def forward(self, batch_s):
            q_eval_s = self.q_values_model.predict(batch_s)
            return q_eval_s

        def learn_batch(self, batch_s, batch_a_indices_one_hot, q_target_chosen_a):
            # print('Training Started')
            self.model.fit([batch_s, batch_a_indices_one_hot], q_target_chosen_a, verbose=0)
            # print('Training Finished')

        def load_model_file(self):
            print("...Loading Keras h5...")
            self.model = keras_models.load_model(self.h5_file)

        def save_model_file(self):
            print("...Saving Keras h5...")
            self.model.save(self.h5_file)

    class NN_Torch(torch.nn.Module):

        def __init__(self, nn, relevant_screen_size, image_channels, device_str='cuda'):
            super(NN.NN_Torch, self).__init__()

            self.nn = nn
            self.relevant_screen_size = relevant_screen_size
            self.image_channels = image_channels

            self.model_file = os.path.join(nn.chkpt_dir, 'q_nn_torch')

            self.build_network()

            self.optimizer = torch_get_optimizer(self.nn.optimizer_type, self.nn.ALPHA, self.parameters())

            self.loss = torch.nn.MSELoss()

            self.device = torch_get_device_according_to_device_type(device_str)
            self.to(self.device)

        def build_network(self):
            if self.nn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                self.fc1 = torch.nn.Linear(*self.nn.input_dims, self.nn.fc_layers_dims[0])
                torch_init.kaiming_normal_(self.fc1.weight.data)
                torch_init.zeros_(self.fc1.bias.data)

                self.fc2 = torch.nn.Linear(self.nn.fc_layers_dims[0], self.nn.fc_layers_dims[1])
                torch_init.kaiming_normal_(self.fc2.weight.data)
                torch_init.zeros_(self.fc2.bias.data)

            else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                frames_stack_size = ATARI_FRAMES_STACK_SIZE
                self.in_channels = frames_stack_size * self.image_channels

                # filters, kernel_size, stride, padding
                conv1_fksp = (32, 8, 4, 1)
                conv2_fksp = (64, 4, 2, 0)
                conv3_fksp = (128, 3, 1, 0)

                i_H, i_W = self.nn.input_dims[0], self.nn.input_dims[1]
                conv1_o_H, conv1_o_W = calc_conv_layer_output_dims(i_H, i_W, *conv1_fksp[1:])
                conv2_o_H, conv2_o_W = calc_conv_layer_output_dims(conv1_o_H, conv1_o_W, *conv2_fksp[1:])
                conv3_o_H, conv3_o_W = calc_conv_layer_output_dims(conv2_o_H, conv2_o_W, *conv3_fksp[1:])
                self.flat_dims = conv3_fksp[0] * conv3_o_H * conv3_o_W

                self.conv1 = torch.nn.Conv2d(self.in_channels, *conv1_fksp)
                torch_init.kaiming_normal_(self.conv1.weight.data)
                torch_init.zeros_(self.conv1.bias.data)
                self.conv1_bn = torch.nn.LayerNorm([conv1_fksp[0], conv1_o_H, conv1_o_W])

                self.conv2 = torch.nn.Conv2d(conv1_fksp[0], *conv2_fksp)
                torch_init.kaiming_normal_(self.conv2.weight.data)
                torch_init.zeros_(self.conv2.bias.data)
                self.conv2_bn = torch.nn.LayerNorm([conv2_fksp[0], conv2_o_H, conv2_o_W])

                self.conv3 = torch.nn.Conv2d(conv2_fksp[0], *conv3_fksp)
                torch_init.kaiming_normal_(self.conv3.weight.data)
                torch_init.zeros_(self.conv3.bias.data)
                self.conv3_bn = torch.nn.LayerNorm([conv3_fksp[0], conv3_o_H, conv3_o_W])

                self.fc1 = torch.nn.Linear(self.flat_dims, self.nn.fc_layers_dims[0])
                torch_init.kaiming_normal_(self.fc1.weight.data)
                torch_init.zeros_(self.fc1.bias.data)

            self.fc_last = torch.nn.Linear(self.nn.fc_layers_dims[-1], self.nn.n_actions)
            torch_init.xavier_normal_(self.fc_last.weight.data)
            torch_init.zeros_(self.fc_last.bias.data)

        def forward(self, batch_s):
            x = torch.tensor(batch_s, dtype=torch.float32).to(self.device)

            if self.nn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                x = torch_func.relu(self.fc1(x))
                x = torch_func.relu(self.fc2(x))

            else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                x = x.view(-1, self.in_channels, *self.relevant_screen_size)
                x = torch_func.relu(self.conv1_bn(self.conv1(x)))
                x = torch_func.relu(self.conv2_bn(self.conv2(x)))
                x = torch_func.relu(self.conv3_bn(self.conv3(x)))
                x = x.view(-1, self.flat_dims).to(self.device)  # Flatten
                x = torch_func.relu(self.fc1(x))

            q_values = self.fc_last(x).to(self.device)
            return q_values

        def learn_batch(self, batch_a_indices, batch_r, batch_terminal,
                        GAMMA, memory_batch_size, q_eval_s, q_eval_s_):

            batch_a_indices = torch.tensor(batch_a_indices, dtype=torch.int64).to(self.device)
            batch_r = torch.tensor(batch_r, dtype=torch.float32).to(self.device)
            batch_terminal = torch.tensor(batch_terminal, dtype=torch.float32).to(self.device)

            q_target = q_eval_s.clone().detach().to(self.device)
            batch_index = torch.tensor(np.arange(memory_batch_size), dtype=torch.int64).to(self.device)
            q_target[batch_index, batch_a_indices] = \
                batch_r + GAMMA * torch.max(q_eval_s_, dim=1)[0] * batch_terminal

            self.optimizer.zero_grad()
            loss = self.loss(q_target, q_eval_s).to(self.device)

            # print('Training Started')
            loss.backward()
            self.optimizer.step()
            # print('Training Finished')

        def load_model_file(self):
            print("...Loading Torch file...")
            self.load_state_dict(torch.load(self.model_file))

        def save_model_file(self):
            print("...Saving Torch file...")
            torch.save(self.state_dict(), self.model_file)


class Agent(object):

    def __init__(self, custom_env, fc_layers_dims, episodes,
                 alpha, optimizer_type=OPTIMIZER_Adam,
                 gamma=None,
                 eps_max=1.0, eps_min=None, eps_dec=None, eps_dec_type=EPS_DEC_LINEAR,
                 memory_size=None, memory_batch_size=None,
                 pure_exploration_phase=0,
                 double_dql=True, tau=10000,
                 lib_type=LIBRARY_TF,
                 base_dir=''):

        self.input_type = custom_env.input_type

        self.GAMMA = gamma if gamma is not None else custom_env.GAMMA
        self.fc_layers_dims = fc_layers_dims

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.action_space = custom_env.action_space

        self.EPS = eps_max
        self.eps_max = eps_max

        if eps_min is not None:
            self.eps_min = eps_min
        elif custom_env.EPS_MIN is not None:
            self.eps_min = custom_env.EPS_MIN
        else:
            self.eps_min = 0.01

        if eps_dec is not None:
            self.eps_dec = eps_dec
        else:
            # will arrive to eps_min after half the episodes:
            self.eps_dec = (self.eps_max - self.eps_min) * 2 / episodes

        self.eps_dec_type = eps_dec_type

        self.pure_exploration_phase = pure_exploration_phase

        self.lib_type = lib_type

        if self.lib_type == LIBRARY_TORCH:
            self.dtype = np.uint8
        else:  # LIBRARY_TF \ LIBRARY_KERAS
            self.dtype = np.int8

        self.memory_size = memory_size if memory_size is not None else custom_env.memory_size
        self.memory_batch_size = memory_batch_size if memory_batch_size is not None else custom_env.memory_batch_size
        self.memory = ReplayBuffer(custom_env, self.memory_size, lib_type, is_discrete_action_space=True)

        self.learn_step_counter = 0

        # sub_dir = get_file_name(None, self, eps=True, replay_buffer=True) + '/'
        sub_dir = ''
        self.chkpt_dir = base_dir + sub_dir
        make_sure_dir_exists(self.chkpt_dir)

        if self.lib_type == LIBRARY_TF:
            tf.reset_default_graph()
        elif self.lib_type == LIBRARY_KERAS:
            keras_backend.clear_session()

        self.policy_nn = self.init_network(custom_env, 'policy')

        if double_dql:
            self.target_nn = self.init_network(custom_env, 'target')
            self.tau = tau
        else:
            self.target_nn = None
            self.tau = None

    def init_network(self, custom_env, name):
        nn_base = NN(custom_env, self.fc_layers_dims, self.optimizer_type, self.ALPHA, self.chkpt_dir)

        if self.lib_type == LIBRARY_TF:
            nn = nn_base.create_nn_tensorflow(name='q_' + name)

        elif self.lib_type == LIBRARY_KERAS:
            nn = nn_base.create_nn_keras()

        else:  # self.lib_type == LIBRARY_TORCH:
            if custom_env.input_type == INPUT_TYPE_STACKED_FRAMES:
                nn = nn_base.create_nn_torch(custom_env.relevant_screen_size, custom_env.image_channels)
            else:
                nn = nn_base.create_nn_torch()

        return nn

    def choose_action(self, s):
        s = s[np.newaxis, :]

        actions_q_values = self.policy_nn.forward(s)[0]
        if self.lib_type == LIBRARY_TORCH:
            a_index = torch.argmax(actions_q_values).item()
        else:  # LIBRARY_TF \ LIBRARY_KERAS
            a_index = np.argmax(actions_q_values)
        a = self.action_space[a_index]

        a = eps_greedy(a, self.EPS, self.action_space)

        return a

    def store_transition(self, s, a, r, s_, is_terminal):
        self.memory.store_transition(s, a, r, s_, is_terminal)

    def learn_wrapper(self):
        if self.target_nn is not None \
                and self.tau is not None \
                and self.learn_step_counter % self.tau == 0:
            self.update_target_network()

        if self.memory.memory_counter >= self.memory_batch_size:
            self.learn()

    def update_target_network(self):
        if self.lib_type == LIBRARY_TF:
            target_network_params = self.target_nn.params
            policy_network_params = self.policy_nn.params
            for t_n_param, p_n_param in zip(target_network_params, policy_network_params):
                self.policy_nn.sess.run(tf.assign(t_n_param, p_n_param))

        elif self.lib_type == LIBRARY_KERAS:
            self.target_nn.model.set_weights(self.policy_nn.model.get_weights())

        else:  # self.lib_type == LIBRARY_TORCH:
            self.target_nn.load_state_dict(self.policy_nn.state_dict())

    def learn(self):
        # print('Learning Session')

        batch_s, batch_s_, batch_r, batch_terminal, batch_a_indices_one_hot, batch_a_indices = \
            self.memory.sample_batch(self.memory_batch_size)

        q_eval_s = self.policy_nn.forward(batch_s)
        q_eval_s_ = self.policy_nn.forward(batch_s_) if self.target_nn is None else self.target_nn.forward(batch_s_)

        if self.lib_type == LIBRARY_TORCH:
            self.policy_nn.learn_batch(batch_a_indices, batch_r, batch_terminal,
                                       self.GAMMA, self.memory_batch_size, q_eval_s, q_eval_s_)
        else:
            q_target_chosen_a = batch_r + self.GAMMA * np.max(q_eval_s_, axis=1) * batch_terminal
            self.policy_nn.learn_batch(batch_s, batch_a_indices_one_hot, q_target_chosen_a)

        self.learn_step_counter += 1

        if self.learn_step_counter > self.pure_exploration_phase:
            self.EPS = decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

    def save_models(self):
        self.policy_nn.save_model_file()
        if self.target_nn is not None:
            self.target_nn.save_model_file()

    def load_models(self):
        self.policy_nn.load_model_file()
        if self.target_nn is not None:
            self.target_nn.load_model_file()


def load_up_agent_memory_with_random_gameplay(custom_env, agent, n_episodes):
    """
    the agent's memory is originally initialized with zeros (which is perfectly acceptable).
    however, we can overwrite these zeros with actual gameplay sampled from the environment.
    """
    if n_episodes is None or n_episodes > custom_env.memory_size:
        n_episodes = custom_env.memory_size

    print('\n', "Loading up the agent's memory with random gameplay.", '\n')

    while agent.memory.memory_counter < n_episodes:
        done = False
        observation = custom_env.env.reset()
        s = custom_env.get_state(observation, None)
        while not done:
            a = np.random.choice(custom_env.action_space)
            observation_, r, done, info = custom_env.env.step(a)
            r = custom_env.update_reward(r, done, info)
            s_ = custom_env.get_state(observation_, s)
            agent.store_transition(s, a, r, s_, done)
            observation, s = observation_, s_

    print('\n', "Done with random gameplay. Game on.", '\n')


def train_agent(custom_env, agent, n_episodes,
                enable_models_saving, load_checkpoint,
                perform_random_gameplay=True, rnd_gameplay_episodes=None,
                visualize=False, record=False):

    scores_history, learn_episode_index, max_avg = load_training_data(agent, load_checkpoint)

    if perform_random_gameplay:
        load_up_agent_memory_with_random_gameplay(custom_env, agent, rnd_gameplay_episodes)

    env = custom_env.env

    if record:
        env = wrappers.Monitor(
            env, 'recordings/DQL/', force=True,
            video_callable=lambda episode_id: episode_id == 0 or episode_id == (n_episodes - 1)
        )

    print('\n', 'Training Started', '\n')
    train_start_time = datetime.datetime.now()

    starting_ep = learn_episode_index + 1
    for i in range(starting_ep, n_episodes):
        ep_start_time = datetime.datetime.now()

        done = False
        ep_score = 0

        observation = env.reset()
        s = custom_env.get_state(observation, None)

        if visualize and i == n_episodes - 1:
            env.render()

        while not done:
            a = agent.choose_action(s)
            observation_, r, done, info = env.step(a)
            r = custom_env.update_reward(r, done, info)
            s_ = custom_env.get_state(observation_, s)
            ep_score += r
            agent.store_transition(s, a, r, s_, done)
            agent.learn_wrapper()
            observation, s = observation_, s_

            if visualize and i == n_episodes - 1:
                env.render()

        scores_history.append(ep_score)
        pickle_save(scores_history, 'scores_history_train_total', agent.chkpt_dir)

        current_avg = print_training_progress(i, ep_score, scores_history, ep_start_time=ep_start_time, eps=agent.EPS)

        if enable_models_saving and current_avg is not None and \
                (max_avg is None or current_avg >= max_avg):
            max_avg = current_avg
            pickle_save(max_avg, 'max_avg', agent.chkpt_dir)
            save_training_data(agent, i, scores_history)

        if visualize and i == n_episodes - 1:
            env.close()

    print('\n', 'Training Ended ~~~ Episodes: %d ~~~ Runtime: %s' %
          (n_episodes - starting_ep, str(datetime.datetime.now() - train_start_time).split('.')[0]), '\n')

    return scores_history
