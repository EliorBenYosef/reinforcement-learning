import datetime
import os
import numpy as np
from gym import wrappers

import tensorflow as tf
import tensorflow_probability as tfp
import keras.models as keras_models
import keras.layers as keras_layers
import keras.initializers as keras_init
import keras.backend as keras_backend
import torch
import torch.nn.functional as torch_func
import torch.nn.init as torch_init
import torch.distributions as torch_dist

from reinforcement_learning.utils.utils import print_training_progress, pickle_save, make_sure_dir_exists,\
     calculate_standardized_returns_of_consecutive_episodes
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH,\
    OPTIMIZER_Adam, INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES, ATARI_FRAMES_STACK_SIZE
from reinforcement_learning.deep_RL.utils.utils import calc_conv_layer_output_dims
from reinforcement_learning.deep_RL.utils.saver_loader import load_training_data, save_training_data
from reinforcement_learning.deep_RL.utils.optimizers import tf_get_optimizer, keras_get_optimizer, torch_get_optimizer
from reinforcement_learning.deep_RL.utils.devices import tf_get_session_according_to_device, \
    torch_get_device_according_to_device_type


class NN(object):
    """
    Policy NN
    """

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, alpha, chkpt_dir):
        self.input_type = custom_env.input_type

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.is_discrete_action_space = custom_env.is_discrete_action_space
        self.n_actions = custom_env.n_actions
        # self.action_space = custom_env.action_space if self.is_discrete_action_space else None
        self.action_boundary = None if self.is_discrete_action_space else custom_env.action_boundary

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
            self.checkpoint_file = os.path.join(nn.chkpt_dir, 'policy_nn_tf.ckpt')

            self.params = tf.trainable_variables(scope=self.name)

        def build_network(self):
            with tf.compat.v1.variable_scope(self.name):
                self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.nn.input_dims], name='s')
                self.G = tf.compat.v1.placeholder(tf.float32, shape=[None], name='G')

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

                if self.nn.is_discrete_action_space:
                    self.a_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name='a_index')

                    a_logits = tf.layers.dense(x, units=self.nn.n_actions,
                                               kernel_initializer=tf.initializers.glorot_normal())
                    self.pi = tf.nn.softmax(a_logits, name='pi')  # a_probs

                    neg_a_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=a_logits, labels=self.a_index)
                    loss = neg_a_log_probs * self.G

                else:
                    self.a_sampled = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nn.n_actions],
                                                              name='a_sampled')

                    self.mu = tf.layers.dense(x, units=self.nn.n_actions, name='mu',  # Mean (μ)
                                              kernel_initializer=tf.initializers.glorot_normal())
                    sigma_unactivated = tf.layers.dense(x, units=self.nn.n_actions, name='sigma_unactivated',  # unactivated STD (σ) - can be a negative number
                                                        kernel_initializer=tf.initializers.glorot_normal())
                    # Element-wise exponential: e^(sigma_unactivated):
                    #   we activate sigma since STD (σ) is strictly real-valued (positive, non-zero - it's not a Dirac delta function).
                    self.sigma = tf.exp(sigma_unactivated, name='sigma')  # STD (σ)

                    gaussian_dist = tfp.distributions.Normal(loc=self.mu, scale=self.sigma)
                    a_log_prob = gaussian_dist.log_prob(self.a_sampled)
                    loss = -tf.reduce_mean(a_log_prob) * self.G

                optimizer = tf_get_optimizer(self.nn.optimizer_type, self.nn.ALPHA)
                self.optimize = optimizer.minimize(loss)  # train_op

        def forward(self, batch_s):
            if self.nn.is_discrete_action_space:
                actor_value = self.sess.run(self.pi, feed_dict={self.s: batch_s})
            else:
                actor_value = self.sess.run([self.mu, self.sigma], feed_dict={self.s: batch_s})
            return actor_value

        def learn_batch(self, memory, memory_G):
            feed_dict = {self.s: np.array(memory.memory_s),
                         self.G: memory_G}
            if self.nn.is_discrete_action_space:
                feed_dict[self.a_index] = np.array(memory.memory_a_indices)
            else:
                feed_dict[self.a_sampled] = np.array(memory.memory_a_sampled)

            # print('Training Started')
            self.sess.run(self.optimize, feed_dict=feed_dict)
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

            self.h5_file = os.path.join(nn.chkpt_dir, 'policy_nn_keras.h5')

            self.build_network()

        def build_network(self):
            s = keras_layers.Input(shape=self.nn.input_dims, dtype='float32', name='s')
            G = keras_layers.Input(shape=(1,), dtype='float32', name='G')

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

            if self.nn.is_discrete_action_space:
                pi = keras_layers.Dense(self.nn.n_actions, activation='softmax', name='pi',  # a_probs = the stochastic policy (π)
                                        kernel_initializer=keras_init.glorot_normal())(x)
                self.policy = keras_models.Model(inputs=s, outputs=pi)
                self.model = keras_models.Model(inputs=[s, G], outputs=pi)  # policy_model
            else:
                mu = keras_layers.Dense(self.nn.n_actions, name='mu',  # Mean (μ)
                                        kernel_initializer=keras_init.glorot_normal())(x)
                sigma_unactivated = keras_layers.Dense(self.nn.n_actions, name='sigma_unactivated',  # unactivated STD (σ) - can be a negative number
                                                       kernel_initializer=keras_init.glorot_normal())(x)
                # Element-wise exponential: e^(sigma_unactivated):
                #   we activate sigma since STD (σ) is strictly real-valued (positive, non-zero - it's not a Dirac delta function).
                sigma = keras_layers.Lambda(lambda sig: keras_backend.exp(sig),  # STD (σ)
                                            name='sigma')(sigma_unactivated)

                self.policy = keras_models.Model(inputs=s, outputs=[mu, sigma])
                self.model = keras_models.Model(inputs=[s, G], outputs=[mu, sigma])  # policy_model

            is_discrete_action_space = self.nn.is_discrete_action_space

            def custom_loss(y_true, y_pred):  # (a_indices_one_hot, actor.output - pi \ [mu, sigma])
                if is_discrete_action_space:
                    prob_chosen_a = keras_backend.sum(y_pred * y_true)  # outputs the prob of the chosen a
                    prob_chosen_a = keras_backend.clip(prob_chosen_a, 1e-8, 1 - 1e-8)  # boundaries to prevent from taking log of 0\1
                    log_prob_chosen_a = keras_backend.log(prob_chosen_a)  # log_probability, negative value (since prob<1)
                    loss = -log_prob_chosen_a * G
                else:
                    mu_pred, sigma_pred = y_pred[0], y_pred[1]  # Mean (μ), STD (σ)
                    gaussian_dist = tfp.distributions.Normal(loc=mu_pred, scale=sigma_pred)
                    a_log_prob = gaussian_dist.log_prob(y_true[0])
                    loss = -keras_backend.mean(a_log_prob) * G

                return loss

            optimizer = keras_get_optimizer(self.nn.optimizer_type, self.nn.ALPHA)
            self.model.compile(optimizer, loss=custom_loss)

        def forward(self, batch_s):
            actor_value = self.policy.predict(batch_s)
            return actor_value

        def learn_batch(self, memory, memory_G):
            memory_s = np.array(memory.memory_s)

            if self.nn.is_discrete_action_space:
                memory_a_indices = np.array(memory.memory_a_indices)
                memory_size = len(memory_a_indices)
                memory_a_indices_one_hot = np.zeros((memory_size, self.nn.n_actions), dtype=np.int8)
                memory_a_indices_one_hot[np.arange(memory_size), memory_a_indices] = 1

                # print('Training Started')
                self.model.fit([memory_s, memory_G], memory_a_indices_one_hot, verbose=0)
                # print('Training Finished')
            else:
                memory_a_sampled = np.array(memory.memory_a_sampled)
                memory_a_sampled = [memory_a_sampled, memory_a_sampled]  # done to match the output's shape

                # print('Training Started')
                self.model.fit([memory_s, memory_G], memory_a_sampled, verbose=0)
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

            self.model_file = os.path.join(nn.chkpt_dir, 'policy_nn_torch')

            self.build_network()

            self.optimizer = torch_get_optimizer(self.nn.optimizer_type, self.nn.ALPHA, self.parameters())

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

            if self.nn.is_discrete_action_space:
                self.pi_unactivated = torch.nn.Linear(self.nn.fc_layers_dims[-1], self.nn.n_actions)
                torch_init.xavier_normal_(self.pi_unactivated.weight.data)
                torch_init.zeros_(self.pi_unactivated.bias.data)

            else:
                self.mu = torch.nn.Linear(self.nn.fc_layers_dims[-1], self.nn.n_actions)
                torch_init.xavier_normal_(self.mu.weight.data)
                torch_init.zeros_(self.mu.bias.data)

                self.sigma_unactivated = torch.nn.Linear(self.nn.fc_layers_dims[-1], self.nn.n_actions)
                torch_init.xavier_normal_(self.sigma_unactivated.weight.data)
                torch_init.zeros_(self.sigma_unactivated.bias.data)

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

            if self.nn.is_discrete_action_space:
                pi = torch_func.softmax(self.pi_unactivated(x))  # a_probs = the stochastic policy (π)
                categorical_dist = torch_dist.Categorical(pi)  # a_probs_dist
                a_sampled = categorical_dist.sample()
                a_log_prob = categorical_dist.log_prob(a_sampled)
                return a_sampled, a_log_prob
            else:
                mu = self.mu(x)  # Mean (μ)
                # sigma = self.sigma_unactivated(x)  # unactivated STD (σ) - can be a negative number
                # STD (σ) is strictly real-valued (positive, non-zero - it's not a Dirac delta function):
                sigma = torch.exp(self.sigma_unactivated(x))  # Element-wise exponential: e^(sigma_unactivated)
                gaussian_dist = torch_dist.Normal(mu, sigma)
                a_sampled = gaussian_dist.sample()
                a_log_prob = gaussian_dist.log_prob(a_sampled)

                a_sampled_act = torch.tanh(a_sampled)
                action_boundary = torch.tensor(self.nn.action_boundary, dtype=torch.float32).to(self.device)
                a = torch.mul(a_sampled_act, action_boundary)
                return a, a_log_prob

        def learn_batch(self, memory, memory_G):
            memory_G = torch.tensor(memory_G, dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()
            loss = 0
            for G, a_log_prob in zip(memory_G, memory.memory_a_log_prob):
                loss += -torch.mean(a_log_prob) * G

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


class Memory(object):

    def __init__(self, custom_env, lib_type):
        self.is_discrete_action_space = custom_env.is_discrete_action_space
        self.n_actions = custom_env.n_actions
        self.action_space = custom_env.action_space if self.is_discrete_action_space else None

        self.lib_type = lib_type

        if self.lib_type == LIBRARY_TORCH:
            self.memory_a_log_prob = []

        else:  # LIBRARY_TF \ LIBRARY_KERAS
            self.memory_s = []
            if self.is_discrete_action_space:
                self.memory_a_indices = []
            else:
                self.memory_a_sampled = []

        self.memory_r = []
        self.memory_terminal = []

    def store_transition(self, s, a, r, is_terminal):
        if self.lib_type != LIBRARY_TORCH:  # LIBRARY_TF \ LIBRARY_KERAS
            self.memory_s.append(s)
            if self.is_discrete_action_space:
                self.memory_a_indices.append(self.action_space.index(a))

        self.memory_r.append(r)
        self.memory_terminal.append(int(is_terminal))

    def store_a_sampled(self, a_sampled):
        self.memory_a_sampled.append(a_sampled)

    def store_a_log_prob(self, a_log_probs):
        self.memory_a_log_prob.append(a_log_probs)

    def reset_memory(self):
        if self.lib_type == LIBRARY_TORCH:
            self.memory_a_log_prob = []

        else:  # LIBRARY_TF \ LIBRARY_KERAS
            self.memory_s = []
            if self.is_discrete_action_space:
                self.memory_a_indices = []
            else:
                self.memory_a_sampled = []

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

        self.is_discrete_action_space = custom_env.is_discrete_action_space
        self.n_actions = custom_env.n_actions
        self.action_space = custom_env.action_space if self.is_discrete_action_space else None
        self.action_boundary = None if self.is_discrete_action_space else custom_env.action_boundary

        self.lib_type = lib_type

        self.memory = Memory(custom_env, lib_type)

        # sub_dir = get_file_name(None, self) + '/'
        sub_dir = ''
        self.chkpt_dir = base_dir + sub_dir
        make_sure_dir_exists(self.chkpt_dir)

        if self.lib_type == LIBRARY_TF:
            tf.reset_default_graph()
        elif self.lib_type == LIBRARY_KERAS:
            keras_backend.clear_session()

        self.policy_nn = self.init_network(custom_env)

    def init_network(self, custom_env):
        nn_base = NN(custom_env, self.fc_layers_dims, self.optimizer_type, self.ALPHA, self.chkpt_dir)

        if self.lib_type == LIBRARY_TF:
            nn = nn_base.create_nn_tensorflow(name='q_policy')

        elif self.lib_type == LIBRARY_KERAS:
            nn = nn_base.create_nn_keras()

        else:  # self.lib_type == LIBRARY_TORCH
            if custom_env.input_type == INPUT_TYPE_STACKED_FRAMES:
                nn = nn_base.create_nn_torch(custom_env.relevant_screen_size, custom_env.image_channels)
            else:
                nn = nn_base.create_nn_torch()

        return nn

    def choose_action(self, s):
        s = s[np.newaxis, :]

        policy_value = self.policy_nn.forward(s)

        if self.is_discrete_action_space:
            a = self.choose_action_discrete(policy_value)
        else:
            a = self.choose_action_continuous(policy_value)

        return a

    def choose_action_discrete(self, policy_value):
        if self.lib_type == LIBRARY_TORCH:
            a_sampled, a_log_prob = policy_value
            self.memory.store_a_log_prob(a_log_prob)
            a_index = a_sampled.item()
            a = self.action_space[a_index]

        else:  # LIBRARY_TF \ LIBRARY_KERAS
            a = np.random.choice(self.action_space, p=policy_value[0])  # pi, a_probs

        return a

    def choose_action_continuous(self, policy_value):
        if self.lib_type == LIBRARY_TORCH:
            a, a_log_prob = policy_value[0][0], policy_value[1][0]
            self.memory.store_a_log_prob(a_log_prob)
            a = np.array([a_t.item() for a_t in a])

        elif self.lib_type == LIBRARY_TF:
            mu, sigma = policy_value[0][0], policy_value[1][0]
            gaussian_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            a_sampled = gaussian_dist.sample()
            a_sampled = a_sampled.eval(session=self.policy_nn.sess)
            self.memory.store_a_sampled(a_sampled)
            a_activated = tf.nn.tanh(a_sampled)
            action_boundary = tf.constant(self.action_boundary, dtype='float32')
            a_tensor = tf.multiply(a_activated, action_boundary)
            a = a_tensor.eval(session=self.policy_nn.sess)

        else:  # LIBRARY_KERAS
            mu, sigma = policy_value[0][0], policy_value[1][0]  # Mean (μ), STD (σ)
            a_sampled = keras_backend.random_normal((1,), mean=mu, stddev=sigma)
            a_sampled = keras_backend.get_value(a_sampled)
            self.memory.store_a_sampled(a_sampled)
            a_activated = keras_backend.tanh(a_sampled)
            action_boundary = keras_backend.constant(self.action_boundary, dtype='float32')
            a_tensor = a_activated * action_boundary
            a = keras_backend.get_value(a_tensor)

        return a

    def store_transition(self, s, a, r, is_terminal):
        self.memory.store_transition(s, a, r, is_terminal)

    def learn(self):
        # print('Learning Session')

        memory_r = np.array(self.memory.memory_r)
        memory_terminal = np.array(self.memory.memory_terminal, dtype=np.int8)
        memory_G = calculate_standardized_returns_of_consecutive_episodes(memory_r, memory_terminal, self.GAMMA)

        self.policy_nn.learn_batch(self.memory, memory_G)
        self.memory.reset_memory()

    def save_models(self):
        self.policy_nn.save_model_file()

    def load_models(self):
        self.policy_nn.load_model_file()


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
            # print('Learn time: %s' % str(datetime.datetime.now() - learn_start_time).split('.')[0])

        if visualize and i == n_episodes - 1:
            env.close()

    print('\n', 'Training Ended ~~~ Episodes: %d ~~~ Runtime: %s' %
          (n_episodes - starting_ep, str(datetime.datetime.now() - train_start_time).split('.')[0]), '\n')

    return scores_history
