import datetime
import os
import numpy as np
from gym import wrappers

import tensorflow as tf
import keras.models as keras_models
import keras.layers as keras_layers
import torch
import torch.nn.functional as torch_func

from reinforcement_learning.utils.utils import print_training_progress, pickle_save, make_sure_dir_exists,\
    decrement_eps, EPS_DEC_LINEAR
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH,\
    OPTIMIZER_Adam, INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES, atari_frames_stack_size
from reinforcement_learning.deep_RL.utils.utils import calc_conv_layer_output_dims
from reinforcement_learning.deep_RL.utils.saver_loader import load_training_data, save_training_data
from reinforcement_learning.deep_RL.utils.optimizers import tf_get_optimizer, keras_get_optimizer, torch_get_optimizer
from reinforcement_learning.deep_RL.utils.devices import tf_get_session_according_to_device, \
    torch_get_device_according_to_device_type
from reinforcement_learning.deep_RL.utils.replay_buffer import ReplayBuffer


class DQN(object):

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, alpha, chkpt_dir):
        self.input_type = custom_env.input_type

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.n_actions = custom_env.n_actions

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.chkpt_dir = chkpt_dir

    def create_dqn_tensorflow(self, name):
        return DQN.DQN_TensorFlow(self, name)

    def create_dqn_keras(self):
        return DQN.DQN_Keras(self)

    def create_dqn_torch(self, relevant_screen_size, image_channels):
        return DQN.DQN_Torch(self, relevant_screen_size, image_channels)

    class DQN_TensorFlow(object):

        def __init__(self, dqn, name, device_map=None):
            self.dqn = dqn

            self.name = name

            self.sess = tf_get_session_according_to_device(device_map)
            self.build_network()
            self.sess.run(tf.compat.v1.global_variables_initializer())

            self.saver = tf.compat.v1.train.Saver()
            self.checkpoint_file = os.path.join(dqn.chkpt_dir, 'dqn_tf.ckpt')

            self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        def build_network(self):
            with tf.compat.v1.variable_scope(self.name):
                self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.dqn.input_dims], name='s')
                self.a_indices_one_hot = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dqn.n_actions],
                                                        name='a_indices_one_hot')
                self.q_target = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dqn.n_actions], name='q_target')

                if self.dqn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                    fc1_ac = tf.layers.dense(inputs=self.s, units=self.dqn.fc_layers_dims[0], activation='relu')
                    fc2_ac = tf.layers.dense(inputs=fc1_ac, units=self.dqn.fc_layers_dims[1], activation='relu')
                    self.q = tf.layers.dense(inputs=fc2_ac, units=self.dqn.n_actions)

                else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                    conv1 = tf.layers.conv2d(inputs=self.s, filters=32,
                                             kernel_size=(8, 8), strides=4, name='conv1',
                                             kernel_initializer=tf.variance_scaling_initializer(scale=2))
                    conv1_ac = tf.nn.relu(conv1)
                    conv2 = tf.layers.conv2d(inputs=conv1_ac, filters=64,
                                             kernel_size=(4, 4), strides=2, name='conv2',
                                             kernel_initializer=tf.variance_scaling_initializer(scale=2))
                    conv2_ac = tf.nn.relu(conv2)
                    conv3 = tf.layers.conv2d(inputs=conv2_ac, filters=128,
                                             kernel_size=(3, 3), strides=1, name='conv3',
                                             kernel_initializer=tf.variance_scaling_initializer(scale=2))
                    conv3_ac = tf.nn.relu(conv3)
                    flat = tf.layers.flatten(conv3_ac)
                    fc1_ac = tf.layers.dense(inputs=flat, units=self.dqn.fc_layers_dims[0], activation='relu',
                                             kernel_initializer=tf.variance_scaling_initializer(scale=2))
                    self.q = tf.layers.dense(inputs=fc1_ac, units=self.dqn.n_actions,
                                             kernel_initializer=tf.variance_scaling_initializer(scale=2.0))
                    # self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions))  # the actual Q value for each action

                self.loss = tf.reduce_mean(tf.square(self.q - self.q_target))  # self.q - self.q_target

                optimizer = tf_get_optimizer(self.dqn.optimizer_type, self.dqn.ALPHA)
                self.optimize = optimizer.minimize(self.loss)  # train_op

        def forward(self, batch_s):
            q_eval_s = self.sess.run(self.q, feed_dict={self.s: batch_s})
            return q_eval_s

        def learn_batch(self, batch_s, batch_a_indices, batch_r, batch_terminal, batch_a_indices_one_hot,
                        input_type, GAMMA, memory_batch_size, q_eval_s, q_eval_s_):

            q_target = q_eval_s.copy()
            batch_index = np.arange(memory_batch_size, dtype=np.int32)
            q_target[batch_index, batch_a_indices] = \
                batch_r + GAMMA * np.max(q_eval_s_, axis=1) * batch_terminal

            # print('Training Started')
            _ = self.sess.run(self.optimize,
                              feed_dict={self.s: batch_s,
                                         self.a_indices_one_hot: batch_a_indices_one_hot,
                                         self.q_target: q_target})
            # print('Training Finished')

        def load_model_file(self):
            print("...Loading TF checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving TF checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class DQN_Keras(object):

        def __init__(self, dqn):
            self.dqn = dqn

            self.h5_file = os.path.join(dqn.chkpt_dir, 'dqn_keras.h5')

            self.optimizer = keras_get_optimizer(self.dqn.optimizer_type, self.dqn.ALPHA)

            self.model = self.build_network()

        def build_network(self):

            if self.dqn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                model = keras_models.Sequential([
                    keras_layers.Dense(self.dqn.fc_layers_dims[0], activation='relu', input_shape=self.dqn.input_dims),
                    keras_layers.Dense(self.dqn.fc_layers_dims[1], activation='relu'),
                    keras_layers.Dense(self.dqn.n_actions)])

            else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                model = keras_models.Sequential([
                    keras_layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation='relu', input_shape=self.dqn.input_dims),
                    keras_layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu'),
                    keras_layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu'),
                    keras_layers.Flatten(),
                    keras_layers.Dense(self.dqn.fc_layers_dims[0], activation='relu'),
                    keras_layers.Dense(self.dqn.n_actions)])

            model.compile(optimizer=self.optimizer, loss='mse')

            return model

        def forward(self, batch_s):
            q_eval_s = self.model.predict(batch_s)
            return q_eval_s

        def learn_batch(self, batch_s, batch_a_indices, batch_r, batch_terminal, batch_a_indices_one_hot,
                        input_type, GAMMA, memory_batch_size, q_eval_s, q_eval_s_):

            q_target = q_eval_s.copy()
            batch_index = np.arange(memory_batch_size, dtype=np.int32)
            q_target[batch_index, batch_a_indices] = \
                batch_r + GAMMA * np.max(q_eval_s_, axis=1) * batch_terminal

            # print('Training Started')
            _ = self.model.fit(batch_s, q_target, verbose=0)
            # print('Training Finished')

        def load_model_file(self):
            print("...Loading Keras h5...")
            self.model = keras_models.load_model(self.h5_file)

        def save_model_file(self):
            print("...Saving Keras h5...")
            self.model.save(self.h5_file)

    class DQN_Torch(torch.nn.Module):

        def __init__(self, dqn, relevant_screen_size, image_channels, device_str='cuda'):

            super(DQN.DQN_Torch, self).__init__()

            self.dqn = dqn
            self.relevant_screen_size = relevant_screen_size
            self.image_channels = image_channels

            self.model_file = os.path.join(dqn.chkpt_dir, 'dqn_torch')

            self.build_network()

            self.optimizer = torch_get_optimizer(self.dqn.optimizer_type, self.parameters(), self.dqn.ALPHA)

            self.loss = torch.nn.MSELoss()

            self.device = torch_get_device_according_to_device_type(device_str)
            self.to(self.device)

        def build_network(self):
            if self.dqn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                self.fc1 = torch.nn.Linear(*self.dqn.input_dims, self.dqn.fc_layers_dims[0])
                self.fc2 = torch.nn.Linear(self.dqn.fc_layers_dims[0], self.dqn.fc_layers_dims[1])
                self.fc3 = torch.nn.Linear(self.dqn.fc_layers_dims[1], self.dqn.n_actions)

            else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                frames_stack_size = atari_frames_stack_size
                self.in_channels = frames_stack_size * self.image_channels

                conv1_filters, conv2_filters, conv3_filters = 32, 64, 128
                conv1_fps = 8, 1, 4
                conv2_fps = 4, 0, 2
                conv3_fps = 3, 0, 1

                self.conv1 = torch.nn.Conv2d(self.in_channels, conv1_filters, conv1_fps[0],
                                       padding=conv1_fps[1], stride=conv1_fps[2])
                self.conv2 = torch.nn.Conv2d(conv1_filters, conv2_filters, conv2_fps[0],
                                       padding=conv2_fps[1], stride=conv2_fps[2])
                self.conv3 = torch.nn.Conv2d(conv2_filters, conv3_filters, conv3_fps[0],
                                       padding=conv3_fps[1], stride=conv3_fps[2])

                i_H, i_W = self.dqn.input_dims[0], self.dqn.input_dims[1]
                conv1_o_H, conv1_o_W = calc_conv_layer_output_dims(i_H, i_W, *conv1_fps)
                conv2_o_H, conv2_o_W = calc_conv_layer_output_dims(conv1_o_H, conv1_o_W, *conv2_fps)
                conv3_o_H, conv3_o_W = calc_conv_layer_output_dims(conv2_o_H, conv2_o_W, *conv3_fps)
                self.flat_dims = conv3_filters * conv3_o_H * conv3_o_W

                self.fc1 = torch.nn.Linear(self.flat_dims, self.dqn.fc_layers_dims[0])
                self.fc2 = torch.nn.Linear(self.dqn.fc_layers_dims[0], self.dqn.n_actions)

        def forward(self, s):
            input = torch.tensor(s, dtype=torch.float).to(self.device)

            if self.dqn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:

                fc1_ac = torch_func.relu(self.fc1(input))
                fc2_ac = torch_func.relu(self.fc2(fc1_ac))
                actions_q_values = self.fc3(fc2_ac)

            else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES

                input = input.view(-1, self.in_channels, *self.relevant_screen_size)
                conv1_ac = torch_func.relu(self.conv1(input))
                conv2_ac = torch_func.relu(self.conv2(conv1_ac))
                conv3_ac = torch_func.relu(self.conv3(conv2_ac))
                flat = conv3_ac.view(-1, self.flat_dims).to(self.device)
                fc1_ac = torch_func.relu(self.fc1(flat))
                actions_q_values = self.fc2(fc1_ac)

            actions_q_values = actions_q_values.to(self.device)

            return actions_q_values

        def learn_batch(self, batch_s, batch_a_indices, batch_r, batch_terminal, batch_a_indices_one_hot,
                        input_type, GAMMA, memory_batch_size, q_eval_s, q_eval_s_):

            batch_index = torch.tensor(np.arange(memory_batch_size), dtype=torch.long).to(self.device)
            batch_a_indices = torch.tensor(batch_a_indices, dtype=torch.long).to(self.device)
            batch_r = torch.tensor(batch_r, dtype=torch.float).to(self.device)
            batch_terminal = torch.tensor(batch_terminal, dtype=torch.float).to(self.device)

            q_target = self.forward(batch_s)  # there's no copy() in torch... need to feed-forward again.
            q_target[batch_index, batch_a_indices] = \
                batch_r + GAMMA * torch.max(q_eval_s_, dim=1)[0] * batch_terminal

            # print('Training Started')
            self.optimizer.zero_grad()
            loss = self.loss(q_target, q_eval_s).to(self.device)
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

        self.policy_dqn = self.init_network(custom_env, 'policy')

        if double_dql:
            self.target_dqn = self.init_network(custom_env, 'target')
            self.tau = tau
        else:
            self.target_dqn = None
            self.tau = None

    def init_network(self, custom_env, name):
        dqn_base = DQN(custom_env, self.fc_layers_dims, self.optimizer_type, self.ALPHA, self.chkpt_dir)

        if self.lib_type == LIBRARY_TF:
            dqn = dqn_base.create_dqn_tensorflow(name='q_' + name)

        elif self.lib_type == LIBRARY_KERAS:
            dqn = dqn_base.create_dqn_keras()

        else:  # self.lib_type == LIBRARY_TORCH:
            if custom_env.input_type == INPUT_TYPE_STACKED_FRAMES:
                relevant_screen_size = custom_env.relevant_screen_size
                image_channels = custom_env.image_channels
            else:
                relevant_screen_size = None
                image_channels = None

            dqn = dqn_base.create_dqn_torch(relevant_screen_size, image_channels)

        return dqn

    def choose_action(self, s):
        s = s[np.newaxis, :]

        actions_q_values = self.policy_dqn.forward(s)
        if self.lib_type == LIBRARY_TORCH:
            action_tensor = torch.argmax(actions_q_values)
            a_index = action_tensor.item()
        else:  # LIBRARY_TF \ LIBRARY_KERAS
            a_index = np.argmax(actions_q_values)
        a = self.action_space[a_index]

        rand = np.random.random()
        if rand < self.EPS:
            # pure exploration, no chance to get the greedy action
            modified_action_space = self.action_space.copy()
            modified_action_space.remove(a)
            a = np.random.choice(modified_action_space)

        return a

    # def choose_action(self, s):
    #     s = s[np.newaxis, :]
    #
    #     rand = np.random.random()
    #     if rand < self.EPS:
    #         # here there's also a chance to get the greedy action
    #         a = np.random.choice(self.action_space)
    #     else:
    #         actions_q_values = self.policy_dqn.forward(s)
    #         if self.lib_type == LIBRARY_TORCH:
    #             action_tensor = torch.argmax(actions_q_values)
    #             a_index = action_tensor.item()
    #         else:  # LIBRARY_TF \ LIBRARY_KERAS
    #             a_index = np.argmax(actions_q_values)
    #         a = self.action_space[a_index]
    #
    #     return a

    def store_transition(self, s, a, r, s_, is_terminal):
        self.memory.store_transition(s, a, r, s_, is_terminal)

    def learn_wrapper(self):
        if self.target_dqn is not None \
                and self.tau is not None \
                and self.learn_step_counter % self.tau == 0:
            self.update_target_network()

        if self.memory.memory_counter >= self.memory_batch_size:
            self.learn()

    def update_target_network(self):
        if self.lib_type == LIBRARY_TF:
            target_network_params = self.target_dqn.params
            policy_network_params = self.policy_dqn.params
            for t_n_param, p_n_param in zip(target_network_params, policy_network_params):
                self.policy_dqn.sess.run(tf.assign(t_n_param, p_n_param))

        elif self.lib_type == LIBRARY_KERAS:
            self.target_dqn.model.set_weights(self.policy_dqn.model.get_weights())

        else:  # self.lib_type == LIBRARY_TORCH:
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    def learn(self):
        # print('Learning Session')

        batch_s, batch_s_, batch_r, batch_terminal, batch_a_indices_one_hot, batch_a_indices = \
            self.memory.sample_batch(self.memory_batch_size)

        q_eval_s = self.policy_dqn.forward(batch_s)
        if self.target_dqn is not None:
            q_eval_s_ = self.target_dqn.forward(batch_s_)
        else:
            q_eval_s_ = self.policy_dqn.forward(batch_s_)

        self.policy_dqn.learn_batch(
            batch_s, batch_a_indices, batch_r, batch_terminal, batch_a_indices_one_hot,
            self.input_type, self.GAMMA, self.memory_batch_size, q_eval_s, q_eval_s_
        )

        self.learn_step_counter += 1

        if self.learn_step_counter > self.pure_exploration_phase:
            self.EPS = decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

    def save_models(self):
        self.policy_dqn.save_model_file()
        if self.target_dqn is not None:
            self.target_dqn.save_model_file()

    def load_models(self):
        self.policy_dqn.load_model_file()
        if self.target_dqn is not None:
            self.target_dqn.load_model_file()


def load_up_agent_memory_with_random_gameplay(custom_env, agent, n_episodes):
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
        # the agent's memory is originally initialized with zeros (which is perfectly acceptable).
        # however, we can overwrite these zeros with actual gameplay sampled from the environment.
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