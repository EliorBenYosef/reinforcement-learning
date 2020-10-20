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
import torch.distributions as torch_dist

from reinforcement_learning.utils.utils import print_training_progress, pickle_save, make_sure_dir_exists,\
     calculate_returns_of_consecutive_episodes
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH,\
    OPTIMIZER_Adam, INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES, atari_frames_stack_size
from reinforcement_learning.deep_RL.utils.utils import calc_conv_layer_output_dims
from reinforcement_learning.deep_RL.utils.saver_loader import load_training_data, save_training_data
from reinforcement_learning.deep_RL.utils.optimizers import tf_get_optimizer, keras_get_optimizer, torch_get_optimizer
from reinforcement_learning.deep_RL.utils.devices import tf_get_session_according_to_device, \
    torch_get_device_according_to_device_type


class DNN(object):

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, alpha, chkpt_dir):
        self.input_type = custom_env.input_type

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.n_actions = custom_env.n_actions

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.chkpt_dir = chkpt_dir

    def create_dnn_tensorflow(self, name):
        return DNN.DNN_TensorFlow(self, name)

    def create_dnn_keras(self):
        return DNN.DNN_Keras(self)

    def create_dnn_torch(self, relevant_screen_size, image_channels):
        return DNN.DNN_Torch(self, relevant_screen_size, image_channels)

    class DNN_TensorFlow(object):

        def __init__(self, dnn, name, device_map=None):
            self.dnn = dnn

            self.name = name

            self.sess = tf_get_session_according_to_device(device_map)
            self.build_network()
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
            self.checkpoint_file = os.path.join(dnn.chkpt_dir, 'dnn_tf.ckpt')

            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        def build_network(self):
            with tf.variable_scope(self.name):
                self.s = tf.placeholder(tf.float32, shape=[None, *self.dnn.input_dims], name='s')
                self.a_index = tf.placeholder(tf.int32, shape=[None], name='a_index')
                self.G = tf.placeholder(tf.float32, shape=[None], name='G')

                if self.dnn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                    fc1_ac = tf.layers.dense(inputs=self.s, units=self.dnn.fc_layers_dims[0],
                                             activation='relu',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    fc2_ac = tf.layers.dense(inputs=fc1_ac, units=self.dnn.fc_layers_dims[1],
                                             activation='relu',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    fc_last = tf.layers.dense(inputs=fc2_ac, units=self.dnn.n_actions,
                                              activation=None,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

                else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                    conv1 = tf.layers.conv2d(inputs=self.s, filters=32,
                                             kernel_size=(8, 8), strides=4, name='conv1',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    conv1_bn = tf.layers.batch_normalization(inputs=conv1, epsilon=1e-5, name='conv1_bn')
                    conv1_bn_ac = tf.nn.relu(conv1_bn)
                    conv2 = tf.layers.conv2d(inputs=conv1_bn_ac, filters=64,
                                             kernel_size=(4, 4), strides=2, name='conv2',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    conv2_bn = tf.layers.batch_normalization(inputs=conv2, epsilon=1e-5, name='conv2_bn')
                    conv2_bn_ac = tf.nn.relu(conv2_bn)
                    conv3 = tf.layers.conv2d(inputs=conv2_bn_ac, filters=128,
                                             kernel_size=(3, 3), strides=1, name='conv3',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    conv3_bn = tf.layers.batch_normalization(inputs=conv3, epsilon=1e-5, name='conv3_bn')
                    conv3_bn_ac = tf.nn.relu(conv3_bn)

                    flat = tf.layers.flatten(conv3_bn_ac)
                    fc1_ac = tf.layers.dense(inputs=flat, units=self.dnn.fc_layers_dims[0],
                                             activation='relu',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    fc_last = tf.layers.dense(inputs=fc1_ac, units=self.dnn.n_actions,
                                              activation=None,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

                self.actions_probabilities = tf.nn.softmax(fc_last, name='actions_probabilities')

                negative_log_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.a_index, logits=fc_last)
                loss = negative_log_probability * self.G
                if self.dnn.input_type == INPUT_TYPE_STACKED_FRAMES:
                    loss = tf.reduce_mean(loss)

                optimizer = tf_get_optimizer(self.dnn.optimizer_type, self.dnn.ALPHA)
                self.optimize = optimizer.minimize(loss)  # train_op

        def get_actions_probabilities(self, batch_s):
            return self.sess.run(self.actions_probabilities, feed_dict={self.s: batch_s})[0]

        def learn_entire_batch(self, memory, GAMMA):
            memory_s = np.array(memory.memory_s)
            memory_a_indices = np.array(memory.memory_a_indices)
            memory_r = np.array(memory.memory_r)
            memory_terminal = np.array(memory.memory_terminal, dtype=np.int8)

            memory_G = calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)

            print('Training Started')
            _ = self.sess.run(self.optimize,
                              feed_dict={self.s: memory_s,
                                         self.a_index: memory_a_indices,
                                         self.G: memory_G})
            print('Training Finished')

        def load_model_file(self):
            print("...Loading TF checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving TF checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class DNN_Keras(object):

        def __init__(self, dnn):
            self.dnn = dnn

            self.h5_file = os.path.join(dnn.chkpt_dir, 'dnn_keras.h5')

            self.optimizer = keras_get_optimizer(self.dnn.optimizer_type, self.dnn.ALPHA)

            self.model, self.policy = self.build_networks()

        def build_networks(self):

            s = keras_layers.Input(shape=self.dnn.input_dims, dtype='float32', name='s')

            if self.dnn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                x = keras_layers.Dense(self.dnn.fc_layers_dims[0], activation='relu',
                                       kernel_initializer=keras_init.glorot_uniform(seed=None))(s)
                x = keras_layers.Dense(self.dnn.fc_layers_dims[1], activation='relu',
                                       kernel_initializer=keras_init.glorot_uniform(seed=None))(x)

            else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES

                x = keras_layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, name='conv1',
                                        kernel_initializer=keras_init.glorot_uniform(seed=None))(s)
                x = keras_layers.BatchNormalization(epsilon=1e-5, name='conv1_bn')(x)
                x = keras_layers.Activation(activation='relu', name='conv1_bn_ac')(x)
                x = keras_layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, name='conv2',
                                        kernel_initializer=keras_init.glorot_uniform(seed=None))(x)
                x = keras_layers.BatchNormalization(epsilon=1e-5, name='conv2_bn')(x)
                x = keras_layers.Activation(activation='relu', name='conv2_bn_ac')(x)
                x = keras_layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, name='conv3',
                                        kernel_initializer=keras_init.glorot_uniform(seed=None))(x)
                x = keras_layers.BatchNormalization(epsilon=1e-5, name='conv3_bn')(x)
                x = keras_layers.Activation(activation='relu', name='conv3_bn_ac')(x)
                x = keras_layers.Flatten()(x)
                x = keras_layers.Dense(self.dnn.fc_layers_dims[0], activation='relu',
                                       kernel_initializer=keras_init.glorot_uniform(seed=None))(x)

            actions_probabilities = keras_layers.Dense(self.dnn.n_actions, activation='softmax', name='actions_probabilities',
                                                       kernel_initializer=keras_init.glorot_uniform(seed=None))(x)

            policy = keras_models.Model(inputs=s, outputs=actions_probabilities)

            #############################

            G = keras_layers.Input(shape=(1,), dtype='float32', name='G')  # advantages. batch_shape=[None]

            def custom_loss(y_true, y_pred):  # (a_indices_one_hot, intermediate_model.output)
                y_pred_clipped = keras_backend.clip(y_pred, 1e-8, 1 - 1e-8)  # we set boundaries so we won't take log of 0\1
                log_lik = y_true * keras_backend.log(y_pred_clipped)  # log_probability
                loss = keras_backend.sum(-log_lik * G)  # keras_backend.mean ?
                return loss

            model = keras_models.Model(inputs=[s, G], outputs=actions_probabilities)  # policy_model
            model.compile(optimizer=self.optimizer, loss=custom_loss)

            return model, policy

        def get_actions_probabilities(self, batch_s):
            actions_probabilities = self.policy.predict(batch_s)[0]
            return actions_probabilities

        def learn_entire_batch(self, memory, GAMMA):
            memory_s = np.array(memory.memory_s)
            memory_a_indices = np.array(memory.memory_a_indices)
            memory_r = np.array(memory.memory_r)
            memory_terminal = np.array(memory.memory_terminal, dtype=np.int8)

            memory_G = calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)

            memory_size = len(memory_a_indices)
            memory_a_indices_one_hot = np.zeros((memory_size, self.dnn.n_actions), dtype=np.int8)
            memory_a_indices_one_hot[np.arange(memory_size), memory_a_indices] = 1

            print('Training Started')
            _ = self.model.fit([memory_s, memory_G], memory_a_indices_one_hot, verbose=0)
            print('Training Finished')

        def load_model_file(self):
            print("...Loading Keras h5...")
            self.model = keras_models.load_model(self.h5_file)

        def save_model_file(self):
            print("...Saving Keras h5...")
            self.model.save(self.h5_file)

    class DNN_Torch(torch.nn.Module):

        def __init__(self, dnn, relevant_screen_size, image_channels, device_str='cuda'):

            super(DNN.DNN_Torch, self).__init__()

            self.dnn = dnn
            self.relevant_screen_size = relevant_screen_size
            self.image_channels = image_channels

            self.model_file = os.path.join(dnn.chkpt_dir, 'dnn_torch')

            self.build_network()

            self.optimizer = torch_get_optimizer(self.dnn.optimizer_type, self.parameters(), self.dnn.ALPHA)

            self.device = torch_get_device_according_to_device_type(device_str)
            self.to(self.device)

        def build_network(self):
            if self.dnn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                self.fc1 = torch.nn.Linear(*self.dnn.input_dims, self.dnn.fc_layers_dims[0])
                self.fc2 = torch.nn.Linear(self.dnn.fc_layers_dims[0], self.dnn.fc_layers_dims[1])
                self.fc3 = torch.nn.Linear(self.dnn.fc_layers_dims[1], self.dnn.n_actions)

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

                i_H, i_W = self.dnn.input_dims[0], self.dnn.input_dims[1]
                conv1_o_H, conv1_o_W = calc_conv_layer_output_dims(i_H, i_W, *conv1_fps)
                conv2_o_H, conv2_o_W = calc_conv_layer_output_dims(conv1_o_H, conv1_o_W, *conv2_fps)
                conv3_o_H, conv3_o_W = calc_conv_layer_output_dims(conv2_o_H, conv2_o_W, *conv3_fps)
                self.flat_dims = conv3_filters * conv3_o_H * conv3_o_W

                self.fc1 = torch.nn.Linear(self.flat_dims, self.dnn.fc_layers_dims[0])
                self.fc2 = torch.nn.Linear(self.dnn.fc_layers_dims[0], self.dnn.n_actions)

        def forward(self, s):
            input = torch.tensor(s, dtype=torch.float).to(self.device)

            if self.dnn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:

                fc1_ac = torch_func.relu(self.fc1(input))
                fc2_ac = torch_func.relu(self.fc2(fc1_ac))
                fc_last = self.fc3(fc2_ac)

            else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES

                input = input.view(-1, self.in_channels, *self.relevant_screen_size)
                conv1_ac = torch_func.relu(self.conv1(input))
                conv2_ac = torch_func.relu(self.conv2(conv1_ac))
                conv3_ac = torch_func.relu(self.conv3(conv2_ac))
                flat = conv3_ac.view(-1, self.flat_dims).to(self.device)
                fc1_ac = torch_func.relu(self.fc1(flat))
                fc_last = self.fc2(fc1_ac)

            actions_probabilities = torch_func.softmax(fc_last).to(self.device)

            return actions_probabilities

        def learn_entire_batch(self, memory, GAMMA):
            memory_a_log_probs = np.array(memory.memory_a_log_probs)
            memory_r = np.array(memory.memory_r)
            memory_terminal = np.array(memory.memory_terminal, dtype=np.uint8)

            memory_G = calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)
            memory_G = torch.tensor(memory_G, dtype=torch.float).to(self.device)

            self.optimizer.zero_grad()
            loss = 0
            for G, a_log_prob in zip(memory_G, memory_a_log_probs):
                loss += -a_log_prob * G
            loss.backward()
            print('Training Started')
            self.optimizer.step()
            print('Training Finished')

        def load_model_file(self):
            print("...Loading Torch file...")
            self.load_state_dict(torch.load(self.model_file))

        def save_model_file(self):
            print("...Saving Torch file...")
            torch.save(self.state_dict(), self.model_file)


class Memory(object):

    def __init__(self, custom_env, lib_type):
        self.n_actions = custom_env.n_actions
        self.action_space = custom_env.action_space

        self.lib_type = lib_type

        if self.lib_type == LIBRARY_TORCH:
            self.memory_a_log_probs = []

        else:  # LIBRARY_TF \ LIBRARY_KERAS
            self.memory_s = []
            self.memory_a_indices = []

        self.memory_r = []
        self.memory_terminal = []

    def store_transition(self, s, a, r, is_terminal):
        if self.lib_type != LIBRARY_TORCH:  # LIBRARY_TF \ LIBRARY_KERAS
            self.memory_s.append(s)
            self.memory_a_indices.append(self.action_space.index(a))

        self.memory_r.append(r)
        self.memory_terminal.append(int(is_terminal))

    def store_a_log_probs(self, a_log_probs):
        if self.lib_type == LIBRARY_TORCH:
            self.memory_a_log_probs.append(a_log_probs)

    def reset_memory(self):
        if self.lib_type == LIBRARY_TORCH:
            self.memory_a_log_probs = []

        else:  # LIBRARY_TF \ LIBRARY_KERAS
            self.memory_s = []
            self.memory_a_indices = []

        self.memory_r = []
        self.memory_terminal = []


class Agent(object):

    def __init__(self, custom_env, fc_layers_dims, ep_batch_num,
                 alpha, optimizer_type=OPTIMIZER_Adam,
                 lib_type=LIBRARY_TF,
                 base_dir=''):

        self.GAMMA = custom_env.GAMMA
        self.fc_layers_dims = fc_layers_dims

        self.ep_batch_num = ep_batch_num

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.action_space = custom_env.action_space

        self.lib_type = lib_type

        self.memory = Memory(custom_env, lib_type)

        # sub_dir = get_file_name(None, self) + '/'
        sub_dir = ''
        self.chkpt_dir = base_dir + sub_dir
        make_sure_dir_exists(self.chkpt_dir)

        self.policy_dnn = self.init_network(custom_env)

    def init_network(self, custom_env):
        dnn_base = DNN(custom_env, self.fc_layers_dims, self.optimizer_type, self.ALPHA, self.chkpt_dir)

        if self.lib_type == LIBRARY_TF:
            dnn = dnn_base.create_dnn_tensorflow(name='q_policy')

        elif self.lib_type == LIBRARY_KERAS:
            dnn = dnn_base.create_dnn_keras()

        else:  # self.lib_type == LIBRARY_TORCH
            if custom_env.input_type == INPUT_TYPE_STACKED_FRAMES:
                relevant_screen_size = custom_env.relevant_screen_size
                image_channels = custom_env.image_channels
            else:
                relevant_screen_size = None
                image_channels = None

            dnn = dnn_base.create_dnn_torch(relevant_screen_size, image_channels)

        return dnn

    def choose_action(self, s):
        s = s[np.newaxis, :]

        if self.lib_type == LIBRARY_TORCH:
            probabilities = self.policy_dnn.forward(s)
            actions_probs = torch_dist.Categorical(probabilities)
            action_tensor = actions_probs.sample()
            a_log_probs = actions_probs.log_prob(action_tensor)
            self.memory.store_a_log_probs(a_log_probs)
            a_index = action_tensor.item()
            a = self.action_space[a_index]

        else:  # LIBRARY_TF \ LIBRARY_KERAS
            probabilities = self.policy_dnn.get_actions_probabilities(s)
            a = np.random.choice(self.action_space, p=probabilities)

        return a

    def store_transition(self, s, a, r, is_terminal):
        self.memory.store_transition(s, a, r, is_terminal)

    def learn(self):
        print('Learning Session')

        self.policy_dnn.learn_entire_batch(self.memory, self.GAMMA)
        self.memory.reset_memory()

    def save_models(self):
        self.policy_dnn.save_model_file()

    def load_models(self):
        self.policy_dnn.load_model_file()


def train_agent(custom_env, agent, n_episodes,
                ep_batch_num,
                enable_models_saving, load_checkpoint,
                visualize=False, record=False):

    scores_history, learn_episode_index, max_avg = load_training_data(agent, load_checkpoint)
    save_episode_index = learn_episode_index

    env = custom_env.env

    if record:
        env = wrappers.Monitor(
            env, 'recordings/PG/', force=True,
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
            agent.store_transition(s, a, r, done)
            observation, s = observation_, s_

            if visualize and i == n_episodes - 1:
                env.render()

        scores_history.append(ep_score)
        pickle_save(scores_history, 'scores_history_train_total', agent.chkpt_dir)

        window = max(ep_batch_num, 100)
        current_avg = print_training_progress(i, ep_score, scores_history, window, ep_start_time=ep_start_time)

        if enable_models_saving and current_avg is not None and learn_episode_index != -1 and \
                (max_avg is None or current_avg >= max_avg):
            max_avg = current_avg
            pickle_save(max_avg, 'max_avg', agent.chkpt_dir)
            if i - save_episode_index - 1 >= ep_batch_num:
                save_episode_index = learn_episode_index
                save_training_data(agent, learn_episode_index, scores_history)

        if (i + 1) % ep_batch_num == 0:
            learn_episode_index = i
            learn_start_time = datetime.datetime.now()
            agent.learn()
            print('Learn time: %s' % str(datetime.datetime.now() - learn_start_time).split('.')[0])

        if visualize and i == n_episodes - 1:
            env.close()

    print('\n', 'Training Ended ~~~ Episodes: %d ~~~ Runtime: %s' %
          (n_episodes - starting_ep, str(datetime.datetime.now() - train_start_time).split('.')[0]), '\n')

    return scores_history
