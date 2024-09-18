import os
import numpy as np

import tensorflow as tf
from tensorflow.python.ops.variables import trainable_variables
import tensorflow_probability.python.distributions as tfpd
import tensorflow.keras.models as keras_models
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.initializers as keras_init
import tensorflow.keras.backend as keras_backend
import torch
import torch.nn.functional as torch_func
import torch.nn.init as torch_init
import torch.distributions as torch_dist

from reinforcement_learning.deep_RL.const import INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES,\
    ATARI_FRAMES_STACK_SIZE
from reinforcement_learning.deep_RL.utils.utils import calc_conv_layer_output_dims
from reinforcement_learning.deep_RL.utils.optimizers import tf_get_optimizer, keras_get_optimizer, torch_get_optimizer
from reinforcement_learning.deep_RL.utils.devices import tf_get_session_according_to_device, \
    torch_get_device_according_to_device_type


# Template

class NN(object):

    def __init__(self, custom_env, fc_layers_dims, optimizers_type_and_lr, chkpt_dir, name_extension):
        self.input_type = custom_env.input_type

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.is_discrete_action_space = custom_env.is_discrete_action_space
        self.n_actions = custom_env.n_actions
        # self.action_space = custom_env.action_space if self.is_discrete_action_space else None
        self.action_boundary = None if self.is_discrete_action_space else custom_env.action_boundary

        self.optimizers_type_and_lr = optimizers_type_and_lr
        # self.optimizer_type = optimizer_type
        # self.ALPHA = alpha; self.lr = lr

        self.chkpt_dir = chkpt_dir
        self.name_extension = name_extension

        # self.name = name
        # self.network_type = network_type
        # self.is_actor = is_actor
        #
        # self.is_discrete_action_space = custom_env.is_discrete_action_space

    def create_nn_tensorflow(self, name):  # self, sess
        return NN.NN_TensorFlow(self, name)  # self, sess

    def create_nn_keras(self):
        return NN.NN_Keras(self)

    def create_nn_torch(self, relevant_screen_size=None, image_channels=None):  # self, device_str='cuda'
        return NN.NN_Torch(self, relevant_screen_size, image_channels)  # self, device_str

    class NN_TensorFlow(object):

        def __init__(self, nn, name, device_map=None):
            self.nn = nn

            self.name = name

            self.sess = tf_get_session_according_to_device(device_map)
            self.build_network()
            self.sess.run(tf.compat.v1.global_variables_initializer())

            self.saver = tf.compat.v1.train.Saver()
            self.checkpoint_file = os.path.join(nn.chkpt_dir, nn.name_extension + 'nn_tf.ckpt')

            self.params = trainable_variables(scope=self.name)
            # self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        def build_network(self):
            with tf.compat.v1.variable_scope(self.name):
                self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.nn.input_dims], name='s')
                # FIRST INSERT here

                if self.nn.input_type == INPUT_TYPE_OBSERVATION_VECTOR:
                    x = tf.compat.v1.layers.dense(self.s, units=self.nn.fc_layers_dims[0], activation='relu',
                                        kernel_initializer=tf.initializers.he_normal())

                else:  # self.input_type == INPUT_TYPE_STACKED_FRAMES
                    x = tf.compat.v1.layers.conv2d(self.s, filters=32, kernel_size=(8, 8), strides=4, name='conv1',
                                         kernel_initializer=tf.initializers.he_normal())
                    x = tf.compat.v1.layers.batch_normalization(x, epsilon=1e-5, name='conv1_bn')
                    x = tf.nn.relu(x)

                    x = tf.compat.v1.layers.conv2d(x, filters=64, kernel_size=(4, 4), strides=2, name='conv2',
                                         kernel_initializer=tf.initializers.he_normal())
                    x = tf.compat.v1.layers.batch_normalization(x, epsilon=1e-5, name='conv2_bn')
                    x = tf.nn.relu(x)

                    x = tf.compat.v1.layers.conv2d(x, filters=128, kernel_size=(3, 3), strides=1, name='conv3',
                                         kernel_initializer=tf.initializers.he_normal())
                    x = tf.compat.v1.layers.batch_normalization(x, epsilon=1e-5, name='conv3_bn')
                    x = tf.nn.relu(x)

                    x = tf.compat.v1.layers.flatten(x)

                x = tf.compat.v1.layers.dense(x, units=self.nn.fc_layers_dims[-1], activation='relu',
                                    kernel_initializer=tf.initializers.he_normal())

                # self.network_output = tf.compat.v1.layers.dense(x, units=self.nn.n_actions, kernel_initializer=)
                # SECOND INSERT here

                optimizer = tf_get_optimizer(*self.nn.optimizers_type_and_lr[0])  # self.nn.optimizer_type, self.nn.ALPHA
                self.optimize = optimizer.minimize(loss)  # train_op

        def forward(self, batch_s):
            pred = self.sess.run(self.network_output, feed_dict={self.s: batch_s})
            return pred

        def learn_batch(self, batch_s, input2, target):
            # print('Training Started')
            self.sess.run(self.optimize,
                          feed_dict={self.s: batch_s,
                                     self.input2: input2,
                                     self.target: target})
            # print('Training Finished')

        def load_model_file(self):
            print("...Loading TF checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving TF checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class NN_Keras(object):

        def __init__(self, nn):  # self, nn, lr_actor, lr_critic
            self.nn = nn

            self.h5_file = os.path.join(nn.chkpt_dir, nn.name_extension + 'nn_keras.h5')
            # self.h5_file_actor = os.path.join(nn.chkpt_dir, nn.name_extension + 'nn_keras_actor.h5')
            # self.h5_file_critic = os.path.join(nn.chkpt_dir, nn.name_extension + 'nn_keras_critic.h5')

            self.build_network()

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

            # INSERT here

            # #############################
            #
            # network_output = keras_layers.Dense(self.nn.n_actions, name='network_output', kernel_initializer=)(x)
            # self.prediction_model = keras_models.Model(inputs=s, outputs=network_output)
            #
            # #############################
            #
            # input2 = keras_layers.Input(shape=(1,), dtype='float32', name='input2')
            #
            # def custom_loss(y_true, y_pred):  # (a_indices_one_hot, a_probs)
            #     prob_chosen_a = keras_backend.sum(y_pred * y_true)
            #     prob_chosen_a = keras_backend.clip(prob_chosen_a, 1e-8, 1 - 1e-8)  # boundaries to prevent from taking log of 0\1
            #     log_prob_chosen_a = keras_backend.log(prob_chosen_a)  # log_probability, negative value (since prob<1)
            #     loss = -log_prob_chosen_a * input2
            #     return loss
            #
            # self.model = keras_models.Model(inputs=[s, input2], outputs=network_output)
            # optimizer = keras_get_optimizer(*self.nn.optimizers_type_and_lr[0])
            # # self.model.compile(optimizer=optimizer, loss='mse')
            # self.model.compile(optimizer=optimizer, loss=custom_loss)

        def forward(self, batch_s):
            pred = self.prediction_model.predict(batch_s)
            return pred

        def learn_batch(self, batch_s, input2, target):
            # print('Training Started')
            self.model.fit([batch_s, input2], target, verbose=0)
            # print('Training Finished')

        def load_model_file(self):
            print("...Loading Keras h5...")
            self.model = keras_models.load_model(self.h5_file)
            # self.actor = keras_models.load_model(self.h5_file_actor)
            # self.critic = keras_models.load_model(self.h5_file_critic)

        def save_model_file(self):
            print("...Saving Keras h5...")
            self.model.save(self.h5_file)
            # self.actor.save(self.h5_file_actor)
            # self.critic.save(self.h5_file_critic)

    class NN_Torch(torch.nn.Module):

        def __init__(self, nn, relevant_screen_size, image_channels, device_str='cuda'):
            super(NN.NN_Torch, self).__init__()

            self.nn = nn
            self.relevant_screen_size = relevant_screen_size
            self.image_channels = image_channels

            self.model_file = os.path.join(nn.chkpt_dir, nn.name_extension + 'nn_torch')

            self.build_network()

            self.optimizer = torch_get_optimizer(*self.nn.optimizers_type_and_lr[0], self.parameters())

            # self.loss = torch.nn.MSELoss()

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

            # INSERT here

        def forward(self, batch_s):  # s
            x = torch.tensor(batch_s, dtype=torch.float32).to(self.device)  # s

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

            # INSERT here

        def learn_batch(self, preds, targets):
            preds = torch.tensor(preds, dtype=torch.float32).to(self.device)
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()
            # loss = self.loss(targets, preds).to(self.device)
            loss = "custom loss here"

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



######################

# Specifics

class DQN:
    """
    Deep Q Learning
    """

    class TensorFlow:

        def build_network(self):
            # FIRST INSERT:
            self.a_indices_one_hot = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nn.n_actions],
                                                              name='a_indices_one_hot')
            self.q_target_chosen_a = tf.compat.v1.placeholder(tf.float32, shape=[None], name='q_target_chosen_a')

            # SECOND INSERT:
            self.q_values = tf.compat.v1.layers.dense(x, units=self.nn.n_actions,
                                            kernel_initializer=tf.initializers.glorot_normal())

            q_chosen_a = tf.reduce_sum(tf.multiply(self.q_values, self.a_indices_one_hot))
            loss = tf.reduce_mean(tf.square(q_chosen_a - self.q_target_chosen_a))  # MSE

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

    class Keras:

        def build_network(self):
            # INSERT:
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

    class Torch:

        def __init__(self):
            self.loss = torch.nn.MSELoss()

        def build_network(self):
            self.fc_last = torch.nn.Linear(self.nn.fc_layers_dims[-1], self.nn.n_actions)
            torch_init.xavier_normal_(self.fc_last.weight.data)
            torch_init.zeros_(self.fc_last.bias.data)

        def forward(self, x):
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


class POLICY_NN:
    """
    Policy Gradient
    """

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
            gaussian_dist = tfpd.Normal(loc=mu, scale=sigma)
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

    class TensorFlow:

        def build_network(self):
            # FIRST INSERT:
            self.G = tf.compat.v1.placeholder(tf.float32, shape=[None], name='G')

            # SECOND INSERT:
            if self.nn.is_discrete_action_space:
                self.a_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name='a_index')

                a_logits = tf.compat.v1.layers.dense(x, units=self.nn.n_actions,
                                           kernel_initializer=tf.initializers.glorot_normal())
                self.pi = tf.nn.softmax(a_logits, name='pi')  # a_probs

                neg_a_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=a_logits, labels=self.a_index)
                loss = neg_a_log_probs * self.G

            else:
                self.a_sampled = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nn.n_actions],
                                                          name='a_sampled')

                self.mu = tf.compat.v1.layers.dense(x, units=self.nn.n_actions, name='mu',  # Mean (μ)
                                          kernel_initializer=tf.initializers.glorot_normal())
                sigma_unactivated = tf.compat.v1.layers.dense(x, units=self.nn.n_actions, name='sigma_unactivated',
                                                    # unactivated STD (σ) - can be a negative number
                                                    kernel_initializer=tf.initializers.glorot_normal())
                # Element-wise exponential: e^(sigma_unactivated):
                #   we activate sigma since STD (σ) is strictly real-valued (positive, non-zero - it's not a Dirac delta function).
                self.sigma = tf.exp(sigma_unactivated, name='sigma')  # STD (σ)

                gaussian_dist = tfpd.Normal(loc=self.mu, scale=self.sigma)
                a_log_prob = gaussian_dist.log_prob(self.a_sampled)
                loss = -tf.reduce_mean(a_log_prob) * self.G

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

    class Keras:

        def build_network(self):
            G = keras_layers.Input(shape=(1,), dtype='float32', name='G')

            # INSERT:
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
                    gaussian_dist = tfpd.Normal(loc=mu_pred, scale=sigma_pred)
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

    class Torch:

        def build_network(self):
            # INSERT:
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

        def forward(self, x):
            # INSERT:
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


class AC_NN:

    class TensorFlow:

        def __init__(self):
            self.a_sampled = None

        def build_network(self):
            # SECOND INSERT:
            if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                self.v_target = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='v_target')

                self.v = tf.compat.v1.layers.dense(x, units=1, activation='linear', name='critic_value',  # Critic layer
                                         kernel_initializer=tf.initializers.glorot_normal())
                critic_loss = tf.square(self.v_target - self.v)  # critic_loss = self.td_error ** 2

            if self.nn.network_type == NETWORK_TYPE_SHARED or self.nn.is_actor:  # Actor layer
                self.td_error = tf.compat.v1.placeholder(tf.float32, shape=[None], name='td_error')

                if self.nn.is_discrete_action_space:
                    self.a_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name='a_index')

                    # self.pi = tf.compat.v1.layers.dense(x, units=self.nn.n_actions, activation='softmax', name='pi',  # a_probs = the stochastic policy (π)
                    #                           kernel_initializer=tf.initializers.glorot_normal())

                    # prob_chosen_a = tf.reduce_sum(tf.multiply(self.pi, self.a_indices_one_hot))  # outputs the prob of the chosen a
                    # prob_chosen_a = tf.clip_by_value(prob_chosen_a, 1e-8, 1 - 1e-8)  # boundaries to prevent from taking log of 0\1
                    # log_prob_chosen_a = tf.log(prob_chosen_a)  # log_probability, negative value (since prob<1)
                    # actor_loss = -log_prob_chosen_a * self.td_error

                    a_logits = tf.compat.v1.layers.dense(x, units=self.nn.n_actions,
                                               kernel_initializer=tf.initializers.glorot_normal())
                    self.pi = tf.nn.softmax(a_logits, name='pi')

                    neg_a_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=a_logits, labels=self.a_index)
                    actor_loss = neg_a_log_probs * self.td_error

                else:
                    self.a_sampled = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nn.n_actions],
                                                              name='a_sampled')

                    self.mu = tf.compat.v1.layers.dense(x, units=self.nn.n_actions, name='mu',  # Mean (μ)
                                              kernel_initializer=tf.initializers.glorot_normal())
                    sigma_unactivated = tf.compat.v1.layers.dense(x, units=self.nn.n_actions, name='sigma_unactivated',
                                                        # unactivated STD (σ) - can be a negative number
                                                        kernel_initializer=tf.initializers.glorot_normal())
                    # Element-wise exponential: e^(sigma_unactivated):
                    #   we activate sigma since STD (σ) is strictly real-valued (positive, non-zero - it's not a Dirac delta function).
                    self.sigma = tf.exp(sigma_unactivated, name='sigma')  # STD (σ)

                    gaussian_dist = tfpd.Normal(loc=self.mu, scale=self.sigma)
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
            return actor_value

        def predict_value(self, s):
            v = self.sess.run(self.v, feed_dict={self.s: s})  # Critic value
            return v

        def train(self, batch_s, v_target=None, td_error=None, a_value=None):
            """
            :param a_value: a_index (Discrete AS), a_sampled (Continuous AS)
            """
            # print('Training Started')
            feed_dict = {self.s: batch_s}

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

        def choose_action(self, s):
            s = s[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                actor_value = self.actor.predict_action(s)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                actor_value = self.actor_critic.predict_action(s)

            if self.ac.is_discrete_action_space:
                a = np.random.choice(self.ac.action_space, p=actor_value[0])
            else:
                a = self.choose_action_continuous(actor_value)

            return a

        def choose_action_continuous(self, actor_value):
            mu, sigma = actor_value[0][0], actor_value[1][0]
            gaussian_dist = tfpd.Normal(loc=mu, scale=sigma)
            a_sampled = gaussian_dist.sample()
            self.a_sampled = a_sampled.eval(session=self.sess)
            a_activated = tf.nn.tanh(self.a_sampled)
            action_boundary = tf.constant(self.ac.action_boundary, dtype='float32')
            a_tensor = tf.multiply(a_activated, action_boundary)
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
                a_index = self.ac.action_space.index(a)
                a_value = np.expand_dims(np.array(a_index), axis=0)
            else:
                a_value = self.a_sampled[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.critic.train(s, v_target=v_target)
                self.actor.train(s, td_error=td_error, a_value=a_value)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.train(s, v_target=v_target, td_error=td_error, a_value=a_value)

    class Keras:

        def __init__(self):
            self.a_sampled = None

        def build_networks(self):
            # INSERT
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

                    self.policy = keras_models.Model(inputs=s, outputs=ms_concat)
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
                        gaussian_dist = tfpd.Normal(loc=mu_pred, scale=sigma_pred)
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

        def learn_batch(self, batch_s, input2, target):
            # print('Training Started')
            self.sess.run(self.optimize,
                          feed_dict={self.s: batch_s,
                                     self.input2: input2,
                                     self.target: target})
            # print('Training Finished')

        def choose_action(self, s):
            s = s[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                actor_value = self.actor.predict_action(s)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                actor_value = self.actor_critic.predict_action(s)

            if self.ac.is_discrete_action_space:
                a = np.random.choice(self.ac.action_space, p=actor_value[0])
            else:
                a = self.choose_action_continuous(actor_value)

            return a

        def choose_action_continuous(self, actor_value):
            mu, sigma = actor_value[0, :self.ac.n_actions], actor_value[0, self.ac.n_actions:]  # Mean (μ), STD (σ)
            a_sampled = keras_backend.random_normal((1,), mean=mu, stddev=sigma)
            self.a_sampled = keras_backend.get_value(a_sampled)
            a_activated = keras_backend.tanh(self.a_sampled)
            action_boundary = keras_backend.constant(self.ac.action_boundary, dtype='float32')
            a_tensor = a_activated * action_boundary
            a = keras_backend.get_value(a_tensor)
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
                a_value = a_indices_one_hot
            else:
                a_value = np.tile(self.a_sampled, reps=2)  # done to match the output's shape

            a_value = a_value[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.actor.fit([s, td_error], a_value, verbose=0)
                self.critic.critic.fit(s, v_target, verbose=0)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.actor.fit([s, td_error], a_value, verbose=0)
                self.actor_critic.critic.fit(s, v_target, verbose=0)

    class Torch:

        def __init__(self):
            self.a_log_prob = None

        def build_network(self):
            # INSERT:
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

            if self.nn.network_type == NETWORK_TYPE_SHARED or not self.nn.is_actor:
                self.v = torch.nn.Linear(self.nn.fc_layers_dims[-1], 1)  # Critic layer
                torch_init.xavier_normal_(self.v.weight.data)
                torch_init.zeros_(self.v.bias.data)

        def forward(self, s, is_actor):
            if is_actor:
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

            if not is_actor:
                v = self.v(x)  # Critic value
                return v

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


class DDPG_NN:

    class NN_TensorFlow(object):

        def __init__(self, custom_env, fc_layers_dims, sess, optimizer_type, lr, name):
            self.name = name

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_actions = custom_env.n_actions

            self.optimizer_type = optimizer_type
            self.lr = lr

            # relevant for the Actor only:
            self.memory_batch_size = custom_env.memory_batch_size
            self.action_boundary = None if custom_env.is_discrete_action_space else \
                custom_env.action_boundary

            self.sess = sess

        def create_actor(self):
            return NN.NN_TF.Actor(self)

        def create_critic(self):
            return NN.NN_TF.Critic(self)

        class Actor(object):

            def __init__(self, ac):
                self.ac = ac

                self.build_network()

                self.params = trainable_variables(scope=self.ac.name)

                self.mu_grads = tf.gradients(self.mu, self.params, -self.a_grads)
                self.normalized_mu_grads = list(map(lambda x: tf.div(x, self.ac.memory_batch_size), self.mu_grads))

                optimizer = tf_get_optimizer(self.ac.optimizer_type, self.ac.lr)
                self.optimize = optimizer.apply_gradients(zip(self.normalized_mu_grads, self.params))  # train_op

            def build_network(self):
                with tf.compat.v1.variable_scope(self.ac.name):
                    self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.ac.input_dims], name='s')
                    self.a_grads = tf.compat.v1.placeholder(tf.float32, shape=[None, self.ac.n_actions], name='a_grads')

                    f1 = 1. / np.sqrt(self.ac.fc_layers_dims[0])
                    x = tf.compat.v1.layers.dense(self.s, units=self.ac.fc_layers_dims[0],
                                        kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                                        bias_initializer=tf.random_uniform_initializer(-f1, f1))
                    x = tf.compat.v1.layers.batch_normalization(x)
                    x = tf.nn.relu(x)

                    f2 = 1. / np.sqrt(self.ac.fc_layers_dims[1])
                    x = tf.compat.v1.layers.dense(x, units=self.ac.fc_layers_dims[1],
                                        kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                                        bias_initializer=tf.random_uniform_initializer(-f2, f2))
                    x = tf.compat.v1.layers.batch_normalization(x)
                    x = tf.nn.relu(x)

                    f3 = 0.003
                    mu = tf.compat.v1.layers.dense(x, units=self.ac.n_actions, activation='tanh',
                                         kernel_initializer=tf.random_uniform_initializer(-f3, f3),
                                         bias_initializer=tf.random_uniform_initializer(-f3, f3))
                    self.mu = tf.multiply(mu, self.ac.action_boundary)  # an ndarray of ndarrays

            def train(self, s, a_grads):
                # print('Training Started')
                self.ac.sess.run(self.optimize,
                                 feed_dict={self.s: s,
                                            self.a_grads: a_grads})
                # print('Training Finished')

            def predict(self, s):
                return self.ac.sess.run(self.mu,
                                        feed_dict={self.s: s})

        class Critic(object):

            def __init__(self, ac):
                self.ac = ac

                self.build_network()

                self.params = trainable_variables(scope=self.ac.name)

                optimizer = tf_get_optimizer(self.ac.optimizer_type, self.ac.lr)
                self.optimize = optimizer.minimize(self.loss)  # train_op

                self.a_grads = tf.gradients(self.q, self.a)  # a list containing an ndarray of ndarrays

            def build_network(self):
                with tf.compat.v1.variable_scope(self.ac.name):
                    self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.ac.input_dims], name='s')
                    self.a = tf.compat.v1.placeholder(tf.float32, shape=[None, self.ac.n_actions], name='a')
                    self.q_target = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='q_target')

                    f1 = 1. / np.sqrt(self.ac.fc_layers_dims[0])
                    x = tf.compat.v1.layers.dense(self.s, units=self.ac.fc_layers_dims[0],
                                        kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                                        bias_initializer=tf.random_uniform_initializer(-f1, f1))
                    x = tf.compat.v1.layers.batch_normalization(x)
                    x = tf.nn.relu(x)

                    f2 = 1. / np.sqrt(self.ac.fc_layers_dims[1])
                    x = tf.compat.v1.layers.dense(x, units=self.ac.fc_layers_dims[1],
                                        kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                                        bias_initializer=tf.random_uniform_initializer(-f2, f2))
                    x = tf.compat.v1.layers.batch_normalization(x)

                    action_in_ac = tf.compat.v1.layers.dense(self.a, units=self.ac.fc_layers_dims[1], activation='relu')

                    state_actions = tf.add(x, action_in_ac)
                    state_actions_ac = tf.nn.relu(state_actions)

                    f3 = 0.003
                    self.q = tf.compat.v1.layers.dense(state_actions_ac, units=1,
                                             kernel_initializer=tf.random_uniform_initializer(-f3, f3),
                                             bias_initializer=tf.random_uniform_initializer(-f3, f3),
                                             kernel_regularizer=tf.keras.regularizers.l2(0.01))

                    self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

            def train(self, s, a, q_target):
                # print('Training Started')
                self.ac.sess.run(self.optimize,
                                 feed_dict={self.s: s,
                                            self.a: a,
                                            self.q_target: q_target})
                # print('Training Finished')

            def predict(self, s, a):
                return self.ac.sess.run(self.q,
                                        feed_dict={self.s: s,
                                                   self.a: a})

            def get_action_gradients(self, inputs, actions):
                return self.ac.sess.run(self.a_grads,
                                        feed_dict={self.s: inputs,
                                                   self.a: actions})

    class NN_Torch(torch.nn.Module):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr, name, chkpt_dir, is_actor, device_str='cuda'):
            super(NN.NN_Torch, self).__init__()

            self.is_actor = is_actor
            self.name = name

            self.model_file = os.path.join(chkpt_dir, 'ddpg_torch_' + name)

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_actions = custom_env.n_actions

            # relevant for the Actor only:
            # self.memory_batch_size = custom_env.memory_batch_size
            self.action_boundary = None if custom_env.is_discrete_action_space else \
                torch.tensor(custom_env.action_boundary, dtype=torch.float32)

            self.build_network()

            self.optimizer = torch_get_optimizer(optimizer_type, lr, self.parameters())

            self.device = torch_get_device_according_to_device_type(device_str)
            self.to(self.device)

        def build_network(self):
            if self.is_actor:
                self.build_network_actor()
            else:
                self.build_network_critic()

        def forward(self, batch_s, batch_a=None):
            if self.is_actor:
                return self.forward_actor(batch_s)
            else:
                return self.forward_critic(batch_s, batch_a)

        def build_network_actor(self):
            f1 = 1. / np.sqrt(self.fc_layers_dims[0])
            self.fc1 = torch.nn.Linear(*self.input_dims, self.fc_layers_dims[0])
            torch_init.uniform_(self.fc1.weight.data, -f1, f1)
            torch_init.uniform_(self.fc1.bias.data, -f1, f1)
            self.fc1_bn = torch.nn.LayerNorm(self.fc_layers_dims[0])

            f2 = 1. / np.sqrt(self.fc_layers_dims[1])
            self.fc2 = torch.nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])
            torch_init.uniform_(self.fc2.weight.data, -f2, f2)
            torch_init.uniform_(self.fc2.bias.data, -f2, f2)
            self.fc2_bn = torch.nn.LayerNorm(self.fc_layers_dims[1])

            f3 = 0.003
            self.mu = torch.nn.Linear(self.fc_layers_dims[1], self.n_actions)
            torch_init.uniform_(self.mu.weight.data, -f3, f3)
            torch_init.uniform_(self.mu.bias.data, -f3, f3)

        def forward_actor(self, batch_s):
            state_value = torch.tensor(batch_s, dtype=torch.float32).to(self.device)

            state_value = self.fc1(state_value)
            state_value = self.fc1_bn(state_value)
            state_value = torch_func.relu(state_value)

            state_value = self.fc2(state_value)
            state_value = self.fc2_bn(state_value)
            state_value = torch_func.relu(state_value)

            mu_value = self.mu(state_value)
            mu_value = torch.tanh(mu_value)
            mu_value = torch.mul(mu_value, self.action_boundary)
            return mu_value.to(self.device)

        def build_network_critic(self):
            f1 = 1. / np.sqrt(self.fc_layers_dims[0])
            self.fc1 = torch.nn.Linear(*self.input_dims, self.fc_layers_dims[0])
            torch_init.uniform_(self.fc1.weight.data, -f1, f1)
            torch_init.uniform_(self.fc1.bias.data, -f1, f1)
            self.fc1_bn = torch.nn.LayerNorm(self.fc_layers_dims[0])

            f2 = 1. / np.sqrt(self.fc_layers_dims[1])
            self.fc2 = torch.nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])
            torch_init.uniform_(self.fc2.weight.data, -f2, f2)
            torch_init.uniform_(self.fc2.bias.data, -f2, f2)
            self.fc2_bn = torch.nn.LayerNorm(self.fc_layers_dims[1])

            self.action_in = torch.nn.Linear(self.n_actions, self.fc_layers_dims[1])

            f3 = 0.003
            self.q = torch.nn.Linear(self.fc_layers_dims[1], 1)
            torch_init.uniform_(self.q.weight.data, -f3, f3)
            torch_init.uniform_(self.q.bias.data, -f3, f3)

            # TODO: add l2 kernel_regularizer of 0.01

        def forward_critic(self, batch_s, batch_a):
            state_value = torch.tensor(batch_s, dtype=torch.float32).to(self.device)

            state_value = self.fc1(state_value)
            state_value = self.fc1_bn(state_value)
            state_value = torch_func.relu(state_value)

            state_value = self.fc2(state_value)
            state_value = self.fc2_bn(state_value)

            action_value = torch.tensor(batch_a, dtype=torch.float32).to(self.device)

            action_value = self.action_in(action_value)
            action_value = torch_func.relu(action_value)

            state_action_value = torch.add(state_value, action_value)
            state_action_value = torch_func.relu(state_action_value)

            q_value = self.q(state_action_value)
            # TODO: apply l2 kernel_regularizer of 0.01
            return q_value.to(self.device)
