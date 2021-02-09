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

from reinforcement_learning.utils.utils import print_training_progress, pickle_save, make_sure_dir_exists
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH,\
    OPTIMIZER_Adam, INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES, ATARI_FRAMES_STACK_SIZE, \
    NETWORK_TYPE_SEPARATE, NETWORK_TYPE_SHARED
from reinforcement_learning.deep_RL.utils.utils import calc_conv_layer_output_dims
from reinforcement_learning.deep_RL.utils.saver_loader import load_training_data, save_training_data
from reinforcement_learning.deep_RL.utils.optimizers import tf_get_optimizer, keras_get_optimizer, torch_get_optimizer
from reinforcement_learning.deep_RL.utils.devices import tf_get_session_according_to_device, \
    torch_get_device_according_to_device_type


class NN(object):

    def __init__(self, custom_env, fc_layers_dims, chkpt_dir, optimizers_type_and_lr, name, network_type, is_actor=False):
        self.input_type = custom_env.input_type

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.is_discrete_action_space = custom_env.is_discrete_action_space
        self.n_actions = custom_env.n_actions
        self.action_space = custom_env.action_space if self.is_discrete_action_space else None
        self.action_boundary = None if self.is_discrete_action_space else custom_env.action_boundary

        self.optimizers_type_and_lr = optimizers_type_and_lr

        self.chkpt_dir = chkpt_dir

        self.name = name
        self.network_type = network_type
        self.is_actor = is_actor

    def create_nn_tensorflow(self, sess):
        return NN.NN_TensorFlow(self, sess)

    def create_nn_keras(self):
        return NN.NN_Keras(self)

    def create_nn_torch(self, relevant_screen_size, image_channels, device_str='cuda'):
        return NN.NN_Torch(self, relevant_screen_size, image_channels, device_str)

    class NN_TensorFlow(object):

        def __init__(self, nn, sess):
            self.nn = nn

            self.sess = sess
            self.build_network()

            self.params = tf.trainable_variables(scope=self.nn.name)

        def build_network(self):
            with tf.compat.v1.variable_scope(self.nn.name):
                self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.nn.input_dims], name='s')

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

                if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                    self.v_target = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='v_target')

                    self.v = tf.layers.dense(x, units=1, activation='linear', name='critic_value',  # Critic layer
                                             kernel_initializer=tf.initializers.glorot_normal())
                    critic_loss = tf.square(self.v_target - self.v)  # critic_loss = self.td_error ** 2

                if self.nn.network_type == NETWORK_TYPE_SHARED or self.nn.is_actor:  # Actor layer
                    self.td_error = tf.compat.v1.placeholder(tf.float32, shape=[None], name='td_error')

                    if self.nn.is_discrete_action_space:
                        self.a_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name='a_index')

                        # self.pi = tf.layers.dense(x, units=self.nn.n_actions, activation='softmax', name='pi',  # a_probs = the stochastic policy (π)
                        #                           kernel_initializer=tf.initializers.glorot_normal())

                        # prob_chosen_a = tf.reduce_sum(tf.multiply(self.pi, self.a_indices_one_hot))  # outputs the prob of the chosen a
                        # prob_chosen_a = tf.clip_by_value(prob_chosen_a, 1e-8, 1 - 1e-8)  # boundaries to prevent from taking log of 0\1
                        # log_prob_chosen_a = tf.log(prob_chosen_a)  # log_probability, negative value (since prob<1)
                        # actor_loss = -log_prob_chosen_a * self.td_error

                        a_logits = tf.layers.dense(x, units=self.nn.n_actions,
                                                   kernel_initializer=tf.initializers.glorot_normal())
                        self.pi = tf.nn.softmax(a_logits, name='pi')

                        neg_a_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=a_logits, labels=self.a_index)
                        actor_loss = neg_a_log_probs * self.td_error

                    else:
                        self.a_sampled = tf.compat.v1.placeholder(tf.float32, shape=[None], name='a_sampled')

                        self.mu = tf.layers.dense(x, units=self.nn.n_actions, name='mu',  # Mean (μ)
                                                  kernel_initializer=tf.initializers.glorot_normal())
                        sigma_unactivated = tf.layers.dense(x, units=self.nn.n_actions, name='sigma_unactivated',  # unactivated STD (σ) - can be a negative number
                                                                 kernel_initializer=tf.initializers.glorot_normal())
                        # Element-wise exponential: e^(sigma_unactivated):
                        #   we activate sigma since STD (σ) is strictly real-valued (positive, non-zero - it's not a Dirac delta function).
                        self.sigma = tf.exp(sigma_unactivated, name='sigma')  # STD (σ)
                        self.ms_concat = tf.concat([self.mu, self.sigma], axis=1)

                        gaussian_dist = tfp.distributions.Normal(loc=self.mu, scale=self.sigma)
                        a_log_prob = gaussian_dist.log_prob(self.a_sampled)
                        actor_loss = -tf.reduce_mean(a_log_prob) * self.td_error

                optimizer = tf_get_optimizer(*self.nn.optimizers_type_and_lr[0])
                loss = critic_loss + actor_loss if self.nn.network_type == NETWORK_TYPE_SHARED \
                    else (actor_loss if self.nn.is_actor else critic_loss)
                self.optimize = optimizer.minimize(loss)  # train_op

        def predict_action(self, s):
            if self.nn.is_discrete_action_space:
                actor_value = self.sess.run(self.pi, feed_dict={self.s: s})  # Actor value
            else:
                actor_value = self.sess.run([self.mu, self.sigma], feed_dict={self.s: s})  # Actor value
                # actor_value = self.sess.run(self.ms_concat, feed_dict={self.s: s})  # Actor value
            return actor_value

        def predict_value(self, s):
            v = self.sess.run(self.v, feed_dict={self.s: s})  # Critic value
            return v

        def train(self, s, v_target=None, td_error=None, a_value=None):
            """
            :param a_value: a_index (Discrete AS), a_sampled (Continuous AS)
            """
            # print('Training Started')
            feed_dict = {self.s: s}

            if self.nn.network_type == NETWORK_TYPE_SHARED or self.nn.is_actor:
                feed_dict[self.td_error] = td_error
                if self.nn.is_discrete_action_space:
                    feed_dict[self.a_index] = a_value
                else:
                    feed_dict[self.a_sampled] = a_value

            if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                feed_dict[self.v_target] = v_target

            self.sess.run(self.optimize, feed_dict=feed_dict)
            # print('Training Finished')

    class NN_Keras(object):

        def __init__(self, nn):
            self.nn = nn

            if self.nn.network_type == NETWORK_TYPE_SHARED or self.nn.is_actor:
                self.h5_file_actor = os.path.join(nn.chkpt_dir, 'ac_nn_keras_actor.h5')
            if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                self.h5_file_critic = os.path.join(nn.chkpt_dir, 'ac_nn_keras_critic.h5')

            self.build_networks()

        def build_networks(self):
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

            if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                critic_value = keras_layers.Dense(1, activation='linear', name='critic_value',
                                                  kernel_initializer=keras_init.glorot_normal())(x)
                self.critic = keras_models.Model(inputs=s, outputs=critic_value)
                optimizer_critic = keras_get_optimizer(*self.nn.optimizers_type_and_lr[1])
                self.critic.compile(optimizer_critic, loss='mse')

            if self.nn.network_type == NETWORK_TYPE_SHARED or self.nn.is_actor:
                td_error = keras_layers.Input(shape=(1,), dtype='float32', name='td_error')

                if self.nn.is_discrete_action_space:
                    pi = keras_layers.Dense(self.nn.n_actions, activation='softmax', name='pi',  # a_probs = the stochastic policy (π)
                                            kernel_initializer=keras_init.glorot_normal())(x)
                    self.policy = keras_models.Model(inputs=s, outputs=pi)
                    self.actor = keras_models.Model(inputs=[s, td_error], outputs=pi)  # policy_model
                else:
                    mu = keras_layers.Dense(self.nn.n_actions, name='mu',  # Mean (μ)
                                            kernel_initializer=keras_init.glorot_normal())(x)
                    sigma_unactivated = keras_layers.Dense(self.nn.n_actions, name='sigma_unactivated',  # unactivated STD (σ) - can be a negative number
                                                           kernel_initializer=keras_init.glorot_normal())(x)
                    # Element-wise exponential: e^(sigma_unactivated):
                    #   we activate sigma since STD (σ) is strictly real-valued (positive, non-zero - it's not a Dirac delta function).
                    sigma = keras_layers.Lambda(lambda sig: keras_backend.exp(sig),  # STD (σ)
                                                name='sigma')(sigma_unactivated)
                    ms_concat = keras_layers.Concatenate(axis=1)([mu, sigma])

                    a_sampled = keras_layers.Lambda(
                        lambda ms: keras_backend.random_normal((1,), mean=ms[0], stddev=ms[1]),
                        name='a_sampled')([mu, sigma])
                    a_activated = keras_layers.Activation('tanh', name='a_activated')(a_sampled)
                    action_boundary = keras_backend.constant(self.nn.action_boundary, dtype='float32')
                    a = keras_layers.Lambda(lambda x: x * action_boundary, name='a')(a_activated)

                    self.policy = keras_models.Model(inputs=s, outputs=a)
                    self.actor = keras_models.Model(inputs=[s, td_error], outputs=ms_concat)  # policy_model

                is_discrete_action_space = self.nn.is_discrete_action_space
                n_actions = self.nn.n_actions

                def actor_loss(y_true, y_pred):  # (a_indices_one_hot, actor.output - pi \ [mu, sigma])
                    if is_discrete_action_space:
                        prob_chosen_a = keras_backend.sum(y_pred * y_true)  # outputs the prob of the chosen a
                        prob_chosen_a = keras_backend.clip(prob_chosen_a, 1e-8, 1 - 1e-8)  # boundaries to prevent from taking log of 0\1
                        log_prob_chosen_a = keras_backend.log(prob_chosen_a)  # log_probability, negative value (since prob<1)
                        loss = -log_prob_chosen_a * td_error
                    else:
                        mu_pred, sigma_pred = y_pred[:n_actions], y_pred[n_actions:]  # Mean (μ), STD (σ)
                        gaussian_dist = tfp.distributions.Normal(loc=mu_pred, scale=sigma_pred)
                        a_log_prob = gaussian_dist.log_prob(y_true[:n_actions])
                        loss = -keras_backend.mean(a_log_prob) * td_error

                    return loss

                optimizer_actor = keras_get_optimizer(*self.nn.optimizers_type_and_lr[0])
                self.actor.compile(optimizer_actor, loss=actor_loss)

        def predict_action(self, s):
            actor_value = self.policy.predict(s)  # Actor value
            return actor_value

        def predict_value(self, s):
            v = self.critic.predict(s)  # Critic value
            return v

        def load_model_file(self):
            print("...Loading Keras h5...")
            if self.nn.network_type == NETWORK_TYPE_SHARED or self.nn.is_actor:
                self.actor = keras_models.load_model(self.h5_file_actor)
            if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                self.critic = keras_models.load_model(self.h5_file_critic)

        def save_model_file(self):
            print("...Saving Keras h5...")
            if self.nn.network_type == NETWORK_TYPE_SHARED or self.nn.is_actor:
                self.actor.save(self.h5_file_actor)
            if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                self.critic.save(self.h5_file_critic)

    class NN_Torch(torch.nn.Module):

        def __init__(self, nn, relevant_screen_size, image_channels, device_str='cuda'):
            super(NN.NN_Torch, self).__init__()

            self.nn = nn
            self.relevant_screen_size = relevant_screen_size
            self.image_channels = image_channels

            self.model_file = os.path.join(nn.chkpt_dir, 'ac_nn_torch_' + self.nn.name)

            self.build_network()

            self.optimizer = torch_get_optimizer(*self.nn.optimizers_type_and_lr[0], self.parameters())

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

            if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                self.v = torch.nn.Linear(self.nn.fc_layers_dims[-1], 1)  # Critic layer
                torch_init.xavier_normal_(self.v.weight.data)
                torch_init.zeros_(self.v.bias.data)

            if self.nn.network_type == NETWORK_TYPE_SHARED or self.nn.is_actor:

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

        def forward(self, batch_s, is_actor):
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

            if not is_actor:
                v = torch_func.linear(self.v(x))  # Critic value
                return v

            if is_actor:
                if self.nn.is_discrete_action_space:
                    pi = torch_func.softmax(self.pi_unactivated(x))  # a_probs = the stochastic policy (π)
                    categorical_dist = torch_dist.Categorical(pi)  # a_probs_dist
                    a_tensor = categorical_dist.sample()
                    a_log_prob = categorical_dist.log_prob(a_tensor)
                    return a_tensor, a_log_prob
                else:
                    mu = self.mu(x)  # Mean (μ)
                    # sigma = self.sigma_unactivated(x)  # unactivated STD (σ) - can be a negative number
                    # STD (σ) is strictly real-valued (positive, non-zero - it's not a Dirac delta function):
                    sigma = torch.exp(self.sigma_unactivated(x))  # Element-wise exponential: e^(sigma_unactivated)
                    gaussian_dist = torch_dist.Normal(mu, sigma)
                    a_tensor = gaussian_dist.sample()
                    a_log_prob = gaussian_dist.log_prob(a_tensor)

                    a_tensor_act = torch.tanh(a_tensor)
                    action_boundary = torch.tensor(self.nn.action_boundary, dtype=torch.float32).to(self.device)
                    a_tensor_act_mul = torch.mul(a_tensor_act, action_boundary)
                    return a_tensor_act_mul, a_log_prob

        def load_model_file(self):
            print("...Loading Torch file...")
            self.load_state_dict(torch.load(self.model_file))

        def save_model_file(self):
            print("...Saving Torch file...")
            torch.save(self.state_dict(), self.model_file)


class AC(object):

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, network_type, chkpt_dir):

        self.GAMMA = custom_env.GAMMA

        self.a_log_prob = None

        self.is_discrete_action_space = custom_env.is_discrete_action_space
        self.n_actions = custom_env.n_actions
        self.action_space = custom_env.action_space if self.is_discrete_action_space else None
        self.action_boundary = None if self.is_discrete_action_space else custom_env.action_boundary

        self.network_type = network_type

        optimizers_type_and_lr = [(optimizer_type, lr_actor), (optimizer_type, lr_critic)]

        if self.network_type == NETWORK_TYPE_SEPARATE:
            self.actor_base = NN(custom_env, fc_layers_dims, chkpt_dir, optimizers_type_and_lr,
                                 'Actor', NETWORK_TYPE_SEPARATE, is_actor=True)
            self.critic_base = NN(custom_env, fc_layers_dims, chkpt_dir, optimizers_type_and_lr,
                                  'Critic', NETWORK_TYPE_SEPARATE, is_actor=False)
        else:  # self.network_type == NETWORK_TYPE_SHARED
            self.actor_critic_base = NN(custom_env, fc_layers_dims, chkpt_dir, optimizers_type_and_lr,
                                        'ActorCritic', NETWORK_TYPE_SHARED)

        self.chkpt_dir = chkpt_dir

    def create_ac_tensorflow(self, device_map):
        return AC.AC_TF(self, device_map)

    def create_ac_keras(self):
        return AC.AC_Keras(self)

    def create_ac_torch(self, custom_env):
        return AC.AC_Torch(self, custom_env)

    class AC_TF(object):

        def __init__(self, ac, device_map):
            self.ac = ac

            self.sess = tf_get_session_according_to_device(device_map)

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor = self.ac.actor_base.create_nn_tensorflow(self.sess)
                self.critic = self.ac.critic_base.create_nn_tensorflow(self.sess)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic = self.ac.actor_critic_base.create_nn_tensorflow(self.sess)

            self.saver = tf.compat.v1.train.Saver()
            self.checkpoint_file = os.path.join(self.ac.chkpt_dir, 'ac_nn_tf.ckpt')

            self.sess.run(tf.compat.v1.global_variables_initializer())

        def choose_action(self, s):
            s = s[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                actor_value = self.actor.predict_action(s)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                actor_value = self.actor_critic.predict_action(s)

            if self.ac.is_discrete_action_space:
                a = np.random.choice(self.ac.action_space, p=actor_value[0])
                return a
            else:
                return self.choose_action_continuous(actor_value)

        def choose_action_continuous(self, actor_value):
            mu, sigma = actor_value[0][0], actor_value[1][0]
            gaussian_dist = tfp.distributions.Normal(loc=mu, scale=sigma)

            self.a_sampled = gaussian_dist.sample()
            a_activated = tf.nn.tanh(self.a_sampled, name='a_activated')
            action_boundary = tf.constant(self.ac.action_boundary, dtype='float32')
            a_tensor = tf.multiply(a_activated, action_boundary, name='a')
            a = a_tensor.eval(session=self.sess)
            return a

        def learn(self, s, a, r, s_, is_terminal):
            # print('Learning Session')
            s = s[np.newaxis, :]
            s_ = s_[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                v = self.critic.predict_value(s)
                v_ = self.critic.predict_value(s_)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                v = self.actor_critic.predict_value(s)
                v_ = self.actor_critic.predict_value(s_)

            v_target = r + self.ac.GAMMA * v_ * (1 - int(is_terminal))
            td_error = np.squeeze(v_target - v, axis=1)

            if self.ac.is_discrete_action_space:
                a_value = self.ac.action_space.index(a)  # a_index
                a_value = np.expand_dims(np.array(a_value), axis=0)
            else:
                a_value = self.a_sampled.eval(session=self.sess)

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.critic.train(s, v_target=v_target)
                self.actor.train(s, td_error=td_error, a_value=a_value)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.train(s, v_target=v_target, td_error=td_error, a_value=a_value)

        def load_model_file(self):
            print("...Loading TF checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving TF checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class AC_Keras(object):

        def __init__(self, ac):
            self.ac = ac

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor = self.ac.actor_base.create_nn_keras()
                self.critic = self.ac.critic_base.create_nn_keras()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic = self.ac.actor_critic_base.create_nn_keras()

        def choose_action(self, s):
            s = s[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                actor_value = self.actor.predict_action(s)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                actor_value = self.actor_critic.predict_action(s)

            if self.ac.is_discrete_action_space:
                a = np.random.choice(self.ac.action_space, p=actor_value[0])
            else:
                a = actor_value[0]

            return a

        def learn(self, s, a, r, s_, is_terminal):
            # print('Learning Session')

            s = s[np.newaxis, :]
            s_ = s_[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                v = self.critic.predict_value(s)
                v_ = self.critic.predict_value(s_)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                v = self.actor_critic.predict_value(s)
                v_ = self.actor_critic.predict_value(s_)

            v_target = r + self.ac.GAMMA * v_ * (1 - int(is_terminal))
            td_error = v_target - v

            if self.ac.is_discrete_action_space:
                a_indices_one_hot = np.zeros(self.ac.n_actions, dtype=np.int8)
                a_index = self.ac.action_space.index(a)
                a_indices_one_hot[a_index] = 1
                a = a_indices_one_hot

            a = a[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.actor.fit(
                    [s, td_error], a if self.ac.is_discrete_action_space else np.tile(a, reps=2), verbose=0)
                self.critic.critic.fit(s, v_target, verbose=0)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.actor.fit(
                    [s, td_error], a if self.ac.is_discrete_action_space else np.tile(a, reps=2), verbose=0)
                self.actor_critic.critic.fit(s, v_target, verbose=0)

        def load_model_file(self):
            print("...Loading Keras models...")
            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.load_model_file()
                self.critic.load_model_file()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.load_model_file()

        def save_model_file(self):
            print("...Saving Keras models...")
            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.save_model_file()
                self.critic.save_model_file()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.save_model_file()

    class AC_Torch(object):

        def __init__(self, ac, custom_env):
            self.ac = ac

            if custom_env.input_type == INPUT_TYPE_STACKED_FRAMES:
                relevant_screen_size = custom_env.relevant_screen_size
                image_channels = custom_env.image_channels
            else:
                relevant_screen_size = None
                image_channels = None

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor = self.ac.actor_base.create_nn_torch(relevant_screen_size, image_channels)
                self.critic = self.ac.critic_base.create_nn_torch(relevant_screen_size, image_channels)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic = self.ac.actor_critic_base.create_nn_torch(relevant_screen_size, image_channels)

        def choose_action(self, s):
            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                actor_value, a_log_prob = self.actor.forward(s, is_actor=True)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                actor_value, a_log_prob = self.actor_critic.forward(s, is_actor=True)

            self.ac.a_log_prob = a_log_prob

            if self.ac.is_discrete_action_space:
                a_index = actor_value.item()
                a = self.ac.action_space[a_index]
            else:
                a = np.array([a_t.item() for a_t in actor_value])

            return a

        def learn(self, s, a, r, s_, is_terminal):
            # print('Learning Session')

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.optimizer.zero_grad()

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                r = torch.tensor(r, dtype=torch.float32).to(self.critic.device)
                v = self.critic.forward(s, is_actor=False)
                v_ = self.critic.forward(s_, is_actor=False)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                r = torch.tensor(r, dtype=torch.float32).to(self.actor_critic.device)
                v = self.actor_critic.forward(s, is_actor=False)
                v_ = self.actor_critic.forward(s_, is_actor=False)

            v_target = r + self.ac.GAMMA * v_ * (1 - int(is_terminal))
            td_error = v_target - v

            actor_loss = -torch.mean(self.ac.a_log_prob) * td_error
            critic_loss = td_error ** 2
            (actor_loss + critic_loss).backward()

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.optimizer.step()

        def load_model_file(self):
            print("...Loading Torch models...")
            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.load_model_file()
                self.critic.load_model_file()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.load_model_file()

        def save_model_file(self):
            print("...Saving Torch models...")
            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.save_model_file()
                self.critic.save_model_file()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.save_model_file()


class Agent(object):

    def __init__(self, custom_env, fc_layers_dims=(400, 300), network_type=NETWORK_TYPE_SHARED,
                 optimizer_type=OPTIMIZER_Adam, lr_actor=0.0001, lr_critic=None,
                 device_type=None, lib_type=LIBRARY_TF,
                 base_dir=''):

        self.GAMMA = custom_env.GAMMA
        self.fc_layers_dims = fc_layers_dims

        self.optimizer_type = optimizer_type
        self.ALPHA = lr_actor
        self.BETA = lr_critic if lr_critic is not None else lr_actor

        # sub_dir = get_file_name(None, self, self.BETA) + '/'
        sub_dir = ''
        self.chkpt_dir = base_dir + sub_dir
        make_sure_dir_exists(self.chkpt_dir)

        if lib_type == LIBRARY_TF:
            tf.reset_default_graph()
        elif lib_type == LIBRARY_KERAS:
            keras_backend.clear_session()

        ac_base = AC(custom_env, fc_layers_dims, optimizer_type, self.ALPHA, self.BETA, network_type, self.chkpt_dir)
        if lib_type == LIBRARY_TF:
            self.ac = ac_base.create_ac_tensorflow(device_type)
        elif lib_type == LIBRARY_KERAS:
            self.ac = ac_base.create_ac_keras()
        else:
            self.ac = ac_base.create_ac_torch(custom_env)

    def choose_action(self, s):
        return self.ac.choose_action(s)

    def learn(self, s, a, r, s_, is_terminal):
        self.ac.learn(s, a, r, s_, is_terminal)

    def save_models(self):
        self.ac.save_model_file()

    def load_models(self):
        self.ac.load_model_file()


def train_agent(custom_env, agent, n_episodes,
                enable_models_saving, load_checkpoint,
                visualize=False, record=False):

    scores_history, learn_episode_index, max_avg = load_training_data(agent, load_checkpoint)

    env = custom_env.env

    if record:
        env = wrappers.Monitor(
            env, 'recordings/AC/', force=True,
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
            agent.learn(s, a, r, s_, done)
            observation, s = observation_, s_

            if visualize and i == n_episodes - 1:
                env.render()

        scores_history.append(ep_score)
        pickle_save(scores_history, 'scores_history_train_total', agent.chkpt_dir)

        current_avg = print_training_progress(i, ep_score, scores_history, ep_start_time=ep_start_time)

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
