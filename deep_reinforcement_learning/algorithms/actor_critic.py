from numpy.random import seed
seed(28)
from tensorflow import set_random_seed
set_random_seed(28)

import os
from gym import wrappers
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import random_uniform_initializer as random_uniform

import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.rmsprop as optim_rmsprop
import torch.optim.adagrad as optim_adagrad
import torch.optim.adadelta as optim_adadelta

import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers

import utils
from deep_reinforcement_learning.envs import Envs


NETWORK_TYPE_SEPARATE = 0
NETWORK_TYPE_SHARED = 1


class DNN(object):

    class AC_DNN_TF(object):

        def __init__(self, custom_env, fc_layers_dims, sess, optimizer_type, lr, name, network_type, is_actor=False):

            self.network_type = network_type
            self.is_actor = is_actor
            self.name = name

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_outputs = custom_env.n_actions if custom_env.is_discrete_action_space else 2

            self.optimizer_type = optimizer_type
            self.lr = lr

            self.sess = sess

            self.build_network()

            self.params = tf.trainable_variables(scope=self.name)

        def build_network(self):
            with tf.variable_scope(self.name):
                self.s = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='s')
                self.td_error = tf.placeholder(tf.float32, shape=[None, 1], name='td_error')
                self.a_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='a_log_probs')
                self.a = tf.placeholder(tf.int32, shape=[None, 1], name='a')

                fc1_activated = tf.layers.dense(inputs=self.s, units=self.fc_layers_dims[0],
                                                activation='relu')
                fc2_activated = tf.layers.dense(inputs=fc1_activated, units=self.fc_layers_dims[1],
                                                activation='relu')

                if self.network_type == NETWORK_TYPE_SEPARATE:  # build_A_or_C_network
                    self.fc3 = tf.layers.dense(inputs=fc2_activated, units=self.n_outputs if self.is_actor else 1)
                    loss = self.get_actor_loss() if self.is_actor else self.get_critic_loss()

                else:  # self.network_type == NETWORK_TYPE_SHARED  # build_A_and_C_network
                    self.fc3 = tf.layers.dense(inputs=fc2_activated, units=self.n_outputs)  # Actor layer
                    self.v = tf.layers.dense(inputs=fc2_activated, units=1)  # Critic layer
                    loss = self.get_actor_loss() + self.get_critic_loss()

                if self.optimizer_type == utils.OPTIMIZER_SGD:
                    optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9)  # SGD + momentum
                    # optimizer = tf.train.GradientDescentOptimizer(self.dqn.ALPHA)  # SGD?
                elif self.optimizer_type == utils.OPTIMIZER_Adagrad:
                    optimizer = tf.train.AdagradOptimizer(self.lr)
                elif self.optimizer_type == utils.OPTIMIZER_Adadelta:
                    optimizer = tf.train.AdadeltaOptimizer(self.lr)
                elif self.optimizer_type == utils.OPTIMIZER_RMSprop:
                    optimizer = tf.train.RMSPropOptimizer(self.lr)
                else:  # self.optimizer_type == utils.OPTIMIZER_Adam
                    optimizer = tf.train.AdamOptimizer(self.lr)

                self.optimize = optimizer.minimize(loss)  # train_op

        def get_actor_loss(self):
            negative_a_log_probs = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.fc3, labels=self.a  # discrete AS: a_index. labels are the actions indices
            )
            actor_loss = negative_a_log_probs * self.td_error
            # if self.dnn.input_type == Envs.INPUT_TYPE_STACKED_FRAMES:
            #     loss = tf.reduce_mean(loss)
            return actor_loss

        # def get_actor_loss(self):
        #     if True:
        #         probabilities = tf.nn.softmax(self.fc3)[0]
        #         actions_probs = tfp.distributions.Categorical(probs=probabilities)
        #         # action_tensor = actions_probs.sample()
        #     else:
        #         mu, sigma_unactivated = self.fc3  # Mean (μ), STD (σ)
        #         sigma = tf.exp(sigma_unactivated)
        #         actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)
        #         # action_tensor = actions_probs.sample(sample_shape=[self.n_actions])
        #
        #     a_log_probs = actions_probs.log_prob(self.a)
        #     # a_log_probs = tf.log(self.fc3[0, self.a])
        #     # # # advantage (TD_error) guided loss:
        #     # # actor_loss = a_log_probs * self.td_error
        #     # # self.exp_v = tf.reduce_mean(actor_loss)
        #
        #     actor_loss = -a_log_probs * self.td_error
        #
        #     return actor_loss

        def get_critic_loss(self):
            critic_loss = self.td_error ** 2
            return critic_loss

        def predict(self, s):
            if self.network_type == NETWORK_TYPE_SEPARATE:
                state_value = self.sess.run(self.fc3, feed_dict={self.s: s})
                return state_value

            else:  # self.network_type == NETWORK_TYPE_SHARED
                state_value = self.sess.run(self.fc3, feed_dict={self.s: s})  # Actor value
                v = self.sess.run(self.v, feed_dict={self.s: s})  # Critic value
                return state_value, v

        def train(self, s, td_error, a=None):
            print('Training Started')
            self.sess.run(self.optimize,
                          feed_dict={self.s: s,
                                     self.td_error: td_error,
                                     self.a: a})
            print('Training Finished')

    class AC_DNN_Torch(nn.Module):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr, name, chkpt_dir, network_type, is_actor=False,
                     device_type='cuda'):
            super(DNN.AC_DNN_Torch, self).__init__()

            self.network_type = network_type
            self.is_actor = is_actor
            self.name = name

            self.model_file = os.path.join(chkpt_dir, 'ac_torch_' + name)

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_outputs = custom_env.n_actions if custom_env.is_discrete_action_space else 2

            self.build_network()

            if optimizer_type == utils.OPTIMIZER_SGD:
                self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
            elif self.optimizer_type == utils.OPTIMIZER_Adagrad:
                self.optimizer = optim_adagrad.Adagrad(self.parameters(), lr=lr)
            elif self.optimizer_type == utils.OPTIMIZER_Adadelta:
                self.optimizer = optim_adadelta.Adadelta(self.parameters(), lr=lr)
            elif optimizer_type == utils.OPTIMIZER_RMSprop:
                self.optimizer = optim_rmsprop.RMSprop(self.parameters(), lr=lr)
            else:  # optimizer_type == utils.OPTIMIZER_Adam
                self.optimizer = optim.Adam(self.parameters(), lr=lr)

            self.device = utils.get_torch_device_according_to_device_type(device_type)
            self.to(self.device)

        def load_model_file(self):
            self.load_state_dict(T.load(self.model_file))

        def save_model_file(self):
            T.save(self.state_dict(), self.model_file)

        def build_network(self):
            self.fc1 = nn.Linear(*self.input_dims, self.fc_layers_dims[0])
            self.fc2 = nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])

            if self.network_type == NETWORK_TYPE_SEPARATE:  # build_A_or_C_network
                self.fc3 = nn.Linear(self.fc_layers_dims[1], self.n_outputs if self.is_actor else 1)

            else:  # self.network_type == NETWORK_TYPE_SHARED    # build_A_and_C_network
                self.fc3 = nn.Linear(self.fc_layers_dims[1], self.n_outputs)  # Actor layer
                self.v = nn.Linear(self.fc_layers_dims[1], 1)  # Critic layer

        def forward(self, s):
            state_value = T.tensor(s, dtype=T.float).to(self.device)

            state_value = self.fc1(state_value)
            state_value = F.relu(state_value)

            state_value = self.fc2(state_value)
            state_value = F.relu(state_value)

            if self.network_type == NETWORK_TYPE_SEPARATE:  # forward_A_or_C_network
                state_value = self.fc3(state_value)
                return state_value

            else:  # self.network_type == NETWORK_TYPE_SHARED    # forward_A_and_C_network
                actor_value = self.fc3(state_value)  # Actor value
                v = self.v(state_value)  # Critic value
                return actor_value, v


class AC(object):

    class AC_TF(object):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, chkpt_dir, network_type, device_type):

            self.GAMMA = custom_env.GAMMA

            # self.a_log_probs = None

            self.is_discrete_action_space = custom_env.is_discrete_action_space
            self.n_actions = custom_env.n_actions
            self.action_space = custom_env.action_space if self.is_discrete_action_space else None
            self.action_boundary = custom_env.action_boundary if not self.is_discrete_action_space else None

            self.sess = utils.get_tf_session_according_to_device_type(device_type)

            self.network_type = network_type
            if self.network_type == NETWORK_TYPE_SEPARATE:
                self.actor = DNN.AC_DNN_TF(
                    custom_env, fc_layers_dims, self.sess, optimizer_type, lr_actor, 'Actor', NETWORK_TYPE_SEPARATE, is_actor=True)
                self.critic = DNN.AC_DNN_TF(
                    custom_env, fc_layers_dims, self.sess, optimizer_type, lr_critic, 'Critic', NETWORK_TYPE_SEPARATE, is_actor=False)

            else:  # self.network_type == NETWORK_TYPE_SHARED
                self.actor_critic = DNN.AC_DNN_TF(
                    custom_env, fc_layers_dims, self.sess, optimizer_type, lr_actor, 'ActorCritic', NETWORK_TYPE_SHARED)

            self.saver = tf.train.Saver()
            self.checkpoint_file = os.path.join(chkpt_dir, 'ac_tf.ckpt')

            self.sess.run(tf.global_variables_initializer())

        def choose_action(self, s):
            s = s[np.newaxis, :]

            if self.network_type == NETWORK_TYPE_SEPARATE:
                actor_value = self.actor.predict(s)
            else:  # self.network_type == NETWORK_TYPE_SHARED
                actor_value, _ = self.actor_critic.predict(s)

            if self.is_discrete_action_space:
                return self.choose_action_discrete(actor_value)
            else:
                return self.choose_action_continuous(actor_value)

        def choose_action_discrete(self, pi):
            probabilities = tf.nn.softmax(pi)[0]
            a_index = np.random.choice(self.action_space, p=probabilities)
            a = self.action_space[a_index]
            return a

        def choose_action_continuous(self, actor_value):
            mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
            sigma = tf.exp(sigma_unactivated)
            actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)
            action_tensor = actions_probs.sample(sample_shape=[self.n_actions])
            action_tensor = tf.nn.tanh(action_tensor)
            action_tensor = tf.multiply(action_tensor, self.action_boundary)
            a = action_tensor.item()
            a = np.array(a).reshape((1,))
            return a

        # def choose_action_discrete(self, pi):
        #     probabilities = tf.nn.softmax(pi)[0]
        #     actions_probs = tfp.distributions.Categorical(probs=probabilities)
        #     action_tensor = actions_probs.sample()
        #     self.a_log_probs = actions_probs.log_prob(action_tensor)
        #     a_index = action_tensor.item()
        #     a = self.action_space[a_index]
        #     return a
        #
        # def choose_action_continuous(self, actor_value):
        #     mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
        #     sigma = tf.exp(sigma_unactivated)
        #     actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)
        #     action_tensor = actions_probs.sample(sample_shape=[self.n_actions])
        #     self.a_log_probs = actions_probs.log_prob(action_tensor)
        #     action_tensor = tf.nn.tanh(action_tensor)
        #     action_tensor = tf.multiply(action_tensor, self.action_boundary)
        #     a = action_tensor.item()
        #     a = np.array(a).reshape((1,))
        #     return a

        def learn(self, s, a, r, s_, is_terminal):
            # print('Learning Session')

            if self.network_type == NETWORK_TYPE_SEPARATE:
                v = self.critic.predict(s)
                v_ = self.critic.predict(s_)
            else:  # self.network_type == NETWORK_TYPE_SHARED
                _, v = self.actor_critic.predict(s)
                _, v_ = self.actor_critic.predict(s_)

            td_error = r + self.GAMMA * v_ * (1 - int(is_terminal)) - v

            if self.is_discrete_action_space:
                a = self.action_space.index(a)

            if self.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.train(s, td_error, a)
                self.critic.train(s, td_error)
            else:  # self.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.train(s, td_error, a)

        def load_model_file(self):
            print("...Loading tf checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving tf checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class AC_Torch(object):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, chkpt_dir, network_type):

            self.GAMMA = custom_env.GAMMA

            self.a_log_probs = None

            self.is_discrete_action_space = custom_env.is_discrete_action_space
            self.n_actions = custom_env.n_actions
            self.action_space = custom_env.action_space if self.is_discrete_action_space else None
            self.action_boundary = custom_env.action_boundary if not self.is_discrete_action_space else None

            self.network_type = network_type
            if self.network_type == NETWORK_TYPE_SEPARATE:
                self.actor = DNN.AC_DNN_Torch(
                    custom_env, fc_layers_dims, optimizer_type, lr_actor, 'Actor', chkpt_dir, network_type, is_actor=True)
                self.critic = DNN.AC_DNN_Torch(
                    custom_env, fc_layers_dims, optimizer_type, lr_critic, 'Critic', chkpt_dir, network_type, is_actor=False)

            else:  # self.network_type == NETWORK_TYPE_SHARED
                self.actor_critic = DNN.AC_DNN_Torch(
                    custom_env, fc_layers_dims, optimizer_type, lr_actor, 'ActorCritic', chkpt_dir, network_type)

        def choose_action(self, s):
            if self.network_type == NETWORK_TYPE_SEPARATE:
                actor_value = self.actor.forward(s)
                device = self.actor.device
            else:  # self.network_type == NETWORK_TYPE_SHARED
                actor_value, _ = self.actor_critic.forward(s)
                device = self.actor_critic.device

            if self.is_discrete_action_space:
                return self.choose_action_discrete(actor_value, device)
            else:
                return self.choose_action_continuous(actor_value, device)

        def choose_action_discrete(self, pi, device):
            probabilities = F.softmax(pi)
            actions_probs = distributions.Categorical(probabilities)
            action_tensor = actions_probs.sample()
            self.a_log_probs = actions_probs.log_prob(action_tensor).to(device)
            a_index = action_tensor.item()
            a = self.action_space[a_index]
            return a

        def choose_action_continuous(self, actor_value, device):
            mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
            sigma = T.exp(sigma_unactivated)
            actions_probs = distributions.Normal(mu, sigma)
            action_tensor = actions_probs.sample(sample_shape=T.Size([self.n_actions]))
            self.a_log_probs = actions_probs.log_prob(action_tensor).to(device)
            action_tensor = T.tanh(action_tensor)
            action_tensor = T.mul(action_tensor, self.action_boundary)
            a = action_tensor.item()
            a = np.array(a).reshape((1,))
            return a

        def learn(self, s, a, r, s_, is_terminal):
            # print('Learning Session')

            if self.network_type == NETWORK_TYPE_SEPARATE:
                r = T.tensor(r, dtype=T.float).to(self.critic.device)
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                v = self.critic.forward(s)
                v_ = self.critic.forward(s_)
            else:  # self.network_type == NETWORK_TYPE_SHARED
                r = T.tensor(r, dtype=T.float).to(self.actor_critic.device)
                self.actor_critic.optimizer.zero_grad()
                _, v = self.actor_critic.forward(s)
                _, v_ = self.actor_critic.forward(s_)

            td_error = r + self.GAMMA * v_ * (1 - int(is_terminal)) - v

            actor_loss = -self.a_log_probs * td_error
            critic_loss = td_error ** 2
            (actor_loss + critic_loss).backward()

            if self.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            else:  # self.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.optimizer.step()

        def load_model_file(self):
            print("...Loading torch models...")
            if self.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.load_model_file()
                self.critic.load_model_file()
            else:  # self.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.load_model_file()

        def save_model_file(self):
            print("...Saving torch models...")
            if self.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.save_model_file()
                self.critic.save_model_file()
            else:  # self.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.save_model_file()


class Agent(object):

    def __init__(self, custom_env, fc_layers_dims=(400, 300),
                 optimizer_type=utils.OPTIMIZER_Adam, lr_actor=0.0001, lr_critic=None,
                 device_type=None, lib_type=utils.LIBRARY_TF):

        self.GAMMA = custom_env.GAMMA
        self.fc_layers_dims = fc_layers_dims

        self.optimizer_type = optimizer_type
        self.ALPHA = lr_actor
        self.BETA = lr_critic if lr_critic is not None else lr_actor

        self.chkpt_dir = 'tmp/' + custom_env.file_name + '/AC/NNs/'

        network_type = NETWORK_TYPE_SEPARATE if lr_critic is not None else NETWORK_TYPE_SHARED
        if lib_type == utils.LIBRARY_TF:
            self.ac = AC.AC_TF(custom_env, fc_layers_dims,
                               optimizer_type, lr_actor, lr_critic,
                               self.chkpt_dir, network_type, device_type)
        else:
            self.ac = AC.AC_Torch(custom_env, fc_layers_dims,
                                  optimizer_type, lr_actor, lr_critic,
                                  self.chkpt_dir, network_type)

    def choose_action(self, s):
        return self.ac.choose_action(s)

    def learn(self, s, a, r, s_, is_terminal):
        self.ac.learn(s, a, r, s_, is_terminal)

    def save_models(self):
        self.ac.save_model_file()

    def load_models(self):
        self.ac.load_model_file()


def train(custom_env, agent, n_episodes,
          enable_models_saving, load_checkpoint, save_checkpoint=25,
          visualize=False, record=False):

    scores_history = []
    episode_index = -1
    if load_checkpoint:
        try:
            agent.load_models()
            print('...Loading episode_index...')
            episode_index = utils.pickle_load('episode_index', agent.chkpt_dir)
            print('...Loading scores_history...')
            scores_history = utils.pickle_load('scores_history', agent.chkpt_dir)
        except (ValueError, tf.OpError):
            print('...No models to load...')
        except FileNotFoundError:
            print('...No data to load...')

    env = custom_env.env

    if record:
        env = wrappers.Monitor(
            env, 'recordings/AC/', force=True,
            video_callable=lambda episode_id: episode_id == 0 or episode_id == (n_episodes - 1)
        )

    print('\n', 'Game Started', '\n')

    for i in range(episode_index + 1, n_episodes):
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
            s_ = custom_env.get_state(observation_, s.copy())
            ep_score += r
            agent.learn(s, a, r, s_, done)
            observation, s = observation_, s_

            if visualize and i == n_episodes - 1:
                env.render()

        scores_history.append(ep_score)

        if enable_models_saving and (i + 1) % save_checkpoint == 0:
            episode_index = i
            utils.pickle_save(episode_index, 'episode_index', agent.chkpt_dir)
            utils.pickle_save(scores_history, 'scores_history', agent.chkpt_dir)
            agent.save_models()

        utils.print_training_progress(i, ep_score, scores_history, custom_env.window)

        if visualize and i == n_episodes - 1:
            env.close()

    print('\n', 'Game Ended', '\n')

    return scores_history


def play(env_type, lib_type=utils.LIBRARY_TORCH, enable_models_saving=False, load_checkpoint=False):
    if lib_type == utils.LIBRARY_KERAS or lib_type == utils.LIBRARY_TF:
        print('\n', "Algorithm currently doesn't work with Keras or TensorFlow", '\n')
        return

    if env_type == 0:
        custom_env = Envs.ClassicControl.CartPole()
        fc_layers_dims = (32, 32)
        optimizer_type = utils.OPTIMIZER_Adam
        alpha = 0.0001  # 0.00001
        beta = 0.0005
        n_episodes = 2500

    elif env_type == 1:
        # custom_env = Envs.Box2D.LunarLander()
        custom_env = Envs.ClassicControl.Pendulum()
        fc_layers_dims = (2048, 512)
        optimizer_type = utils.OPTIMIZER_Adam
        alpha = 0.00001
        beta = None
        n_episodes = 2000

    else:
        custom_env = Envs.ClassicControl.MountainCarContinuous()
        fc_layers_dims = (256, 256)
        optimizer_type = utils.OPTIMIZER_Adam
        alpha = 0.000005
        beta = 0.00001
        n_episodes = 100  # longer than 100 --> instability (because the value function estimation is unstable)

    if custom_env.input_type != Envs.INPUT_TYPE_OBSERVATION_VECTOR:
        print('\n', 'Algorithm currently works only with INPUT_TYPE_OBSERVATION_VECTOR!', '\n')
        return

    custom_env.env.seed(28)

    if lib_type == utils.LIBRARY_TF:
        utils.tf_set_device()

    agent = Agent(custom_env, fc_layers_dims, optimizer_type, lr_actor=alpha, lr_critic=beta, lib_type=lib_type)

    scores_history = train(custom_env, agent, n_episodes, enable_models_saving, load_checkpoint)

    utils.plot_running_average(
        custom_env.name, 'AC', scores_history, window=custom_env.window, show=False,
        file_name=utils.get_plot_file_name(custom_env.file_name, agent, beta=agent.BETA)
    )


if __name__ == '__main__':
    play(0, lib_type=utils.LIBRARY_TORCH)          # CartPole (0), Pendulum (1), MountainCarContinuous (2)
