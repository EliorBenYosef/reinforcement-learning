"""
https://arxiv.org/pdf/1509.02971.pdf
"""

import datetime
import os
import numpy as np
from gym import wrappers

import tensorflow as tf
from tensorflow.python import random_uniform_initializer as tf_init_uni
import torch
import torch.nn.functional as torch_func

from reinforcement_learning.utils.utils import print_training_progress, pickle_save, make_sure_dir_exists
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH,\
    OPTIMIZER_Adam, INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES, atari_frames_stack_size
from reinforcement_learning.deep_RL.utils.saver_loader import load_training_data, save_training_data
from reinforcement_learning.deep_RL.utils.optimizers import tf_get_optimizer, keras_get_optimizer, torch_get_optimizer
from reinforcement_learning.deep_RL.utils.devices import tf_get_session_according_to_device, \
    torch_get_device_according_to_device_type
from reinforcement_learning.deep_RL.utils.replay_buffer import ReplayBuffer


class DNN:

    class AC_DNN_TF(object):

        def __init__(self, custom_env, fc_layers_dims, sess, optimizer_type, lr, name):
            self.name = name

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_actions = custom_env.n_actions

            self.optimizer_type = optimizer_type
            self.lr = lr

            # relevant for the Actor only:
            self.memory_batch_size = custom_env.memory_batch_size
            self.action_boundary = custom_env.action_boundary if not custom_env.is_discrete_action_space \
                else None

            self.sess = sess

        def create_actor(self):
            return DNN.AC_DNN_TF.Actor(self)

        def create_critic(self):
            return DNN.AC_DNN_TF.Critic(self)

        class Actor(object):

            def __init__(self, ac):
                self.ac = ac

                self.build_network()

                self.params = tf.trainable_variables(scope=self.ac.name)

                self.mu_gradients = tf.gradients(self.mu, self.params, -self.a_grad)
                self.normalized_mu_gradients = list(
                    map(lambda x: tf.div(x, self.ac.memory_batch_size), self.mu_gradients))

                optimizer = tf_get_optimizer(self.ac.optimizer_type, self.ac.lr)
                self.optimize = optimizer.apply_gradients(zip(self.normalized_mu_gradients, self.params))  # train_op

            def build_network(self):
                with tf.variable_scope(self.ac.name):
                    self.s = tf.placeholder(tf.float32, shape=[None, *self.ac.input_dims], name='s')
                    self.a_grad = tf.placeholder(tf.float32, shape=[None, self.ac.n_actions], name='a_grad')

                    f1 = 1. / np.sqrt(self.ac.fc_layers_dims[0])
                    fc1 = tf.layers.dense(inputs=self.s, units=self.ac.fc_layers_dims[0],
                                          kernel_initializer=tf_init_uni(-f1, f1),
                                          bias_initializer=tf_init_uni(-f1, f1))
                    fc1_bn = tf.layers.batch_normalization(fc1)
                    fc1_bn_ac = tf.nn.relu(fc1_bn)

                    f2 = 1. / np.sqrt(self.ac.fc_layers_dims[1])
                    fc2 = tf.layers.dense(inputs=fc1_bn_ac, units=self.ac.fc_layers_dims[1],
                                          kernel_initializer=tf_init_uni(-f2, f2),
                                          bias_initializer=tf_init_uni(-f2, f2))
                    fc2_bn = tf.layers.batch_normalization(fc2)
                    fc2_bn_ac = tf.nn.relu(fc2_bn)

                    f3 = 0.003
                    mu = tf.layers.dense(inputs=fc2_bn_ac, units=self.ac.n_actions,
                                         activation='tanh',
                                         kernel_initializer=tf_init_uni(-f3, f3),
                                         bias_initializer=tf_init_uni(-f3, f3))
                    self.mu = tf.multiply(mu, self.ac.action_boundary)  # an ndarray of ndarrays

            def train(self, s, a_grad):
                # print('Training Started')
                self.ac.sess.run(self.optimize,
                                 feed_dict={self.s: s,
                                            self.a_grad: a_grad})
                # print('Training Finished')

            def predict(self, s):
                return self.ac.sess.run(self.mu,
                                        feed_dict={self.s: s})

        class Critic(object):

            def __init__(self, ac):
                self.ac = ac

                self.build_network()

                self.params = tf.trainable_variables(scope=self.ac.name)

                optimizer = tf_get_optimizer(self.ac.optimizer_type, self.ac.lr)
                self.optimize = optimizer.minimize(self.loss)  # train_op

                self.action_gradients = tf.gradients(self.q, self.a)  # a list containing an ndarray of ndarrays

            def build_network(self):
                with tf.variable_scope(self.ac.name):
                    self.s = tf.placeholder(tf.float32, shape=[None, *self.ac.input_dims], name='s')
                    self.a = tf.placeholder(tf.float32, shape=[None, self.ac.n_actions], name='a')
                    self.q_target = tf.placeholder(tf.float32, shape=[None, 1], name='q_target')

                    f1 = 1. / np.sqrt(self.ac.fc_layers_dims[0])
                    fc1 = tf.layers.dense(inputs=self.s, units=self.ac.fc_layers_dims[0],
                                          kernel_initializer=tf_init_uni(-f1, f1),
                                          bias_initializer=tf_init_uni(-f1, f1))
                    fc1_bn = tf.layers.batch_normalization(fc1)
                    fc1_bn_ac = tf.nn.relu(fc1_bn)

                    f2 = 1. / np.sqrt(self.ac.fc_layers_dims[1])
                    fc2 = tf.layers.dense(inputs=fc1_bn_ac, units=self.ac.fc_layers_dims[1],
                                          kernel_initializer=tf_init_uni(-f2, f2),
                                          bias_initializer=tf_init_uni(-f2, f2))
                    fc2_bn = tf.layers.batch_normalization(fc2)

                    action_in_ac = tf.layers.dense(inputs=self.a, units=self.ac.fc_layers_dims[1],
                                                   activation='relu')

                    state_actions = tf.add(fc2_bn, action_in_ac)
                    state_actions_ac = tf.nn.relu(state_actions)

                    f3 = 0.003
                    self.q = tf.layers.dense(inputs=state_actions_ac, units=1,
                                             kernel_initializer=tf_init_uni(-f3, f3),
                                             bias_initializer=tf_init_uni(-f3, f3),
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
                return self.ac.sess.run(self.action_gradients,
                                        feed_dict={self.s: inputs,
                                                   self.a: actions})

    class AC_DNN_Torch(torch.nn.Module):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr, name, chkpt_dir, is_actor, device_str='cuda'):
            super(DNN.AC_DNN_Torch, self).__init__()

            self.is_actor = is_actor
            self.name = name

            self.model_file = os.path.join(chkpt_dir, 'ddpg_torch_' + name)

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_actions = custom_env.n_actions

            # relevant for the Actor only:
            # self.memory_batch_size = custom_env.memory_batch_size
            self.action_boundary = custom_env.action_boundary if not custom_env.is_discrete_action_space \
                else None

            self.build_network()

            self.optimizer = torch_get_optimizer(optimizer_type, self.parameters(), lr)

            self.device = torch_get_device_according_to_device_type(device_str)
            self.to(self.device)

        def load_model_file(self):
            self.load_state_dict(torch.load(self.model_file))

        def save_model_file(self):
            torch.save(self.state_dict(), self.model_file)

        def build_network(self):
            if self.is_actor:
                self.build_network_actor()
            else:
                self.build_network_critic()

        def forward(self, s, a=None):
            if self.is_actor:
                return self.forward_actor(s)
            else:
                return self.forward_critic(s, a)

        def build_network_actor(self):
            f1 = 1. / np.sqrt(self.fc_layers_dims[0])
            self.fc1 = torch.nn.Linear(*self.input_dims, self.fc_layers_dims[0])
            torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
            torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
            self.fc1_bn = torch.nn.LayerNorm(self.fc_layers_dims[0])

            f2 = 1. / np.sqrt(self.fc_layers_dims[1])
            self.fc2 = torch.nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])
            torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
            torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
            self.fc2_bn = torch.nn.LayerNorm(self.fc_layers_dims[1])

            f3 = 0.003
            self.mu = torch.nn.Linear(self.fc_layers_dims[1], self.n_actions)
            torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
            torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        def forward_actor(self, s):
            state_value = torch.tensor(s, dtype=torch.float).to(self.device)

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
            torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
            torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
            self.fc1_bn = torch.nn.LayerNorm(self.fc_layers_dims[0])

            f2 = 1. / np.sqrt(self.fc_layers_dims[1])
            self.fc2 = torch.nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])
            torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
            torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
            self.fc2_bn = torch.nn.LayerNorm(self.fc_layers_dims[1])

            self.action_in = torch.nn.Linear(self.n_actions, self.fc_layers_dims[1])

            f3 = 0.003
            self.q = torch.nn.Linear(self.fc_layers_dims[1], 1)
            torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
            torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

            # TODO: add l2 kernel_regularizer of 0.01

        def forward_critic(self, s, a):
            state_value = torch.tensor(s, dtype=torch.float).to(self.device)

            state_value = self.fc1(state_value)
            state_value = self.fc1_bn(state_value)
            state_value = torch_func.relu(state_value)

            state_value = self.fc2(state_value)
            state_value = self.fc2_bn(state_value)

            action_value = torch.tensor(a, dtype=torch.float).to(self.device)

            action_value = self.action_in(action_value)
            action_value = torch_func.relu(action_value)

            state_action_value = torch.add(state_value, action_value)
            state_action_value = torch_func.relu(state_action_value)

            q_value = self.q(state_action_value)
            # TODO: apply l2 kernel_regularizer of 0.01
            return q_value.to(self.device)


class AC(object):

    class AC_TF(object):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, tau, chkpt_dir, device_map):

            self.GAMMA = 0.99
            self.TAU = tau

            self.sess = tf_get_session_according_to_device(device_map)

            #############################

            # Networks:

            self.actor = DNN.AC_DNN_TF(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_actor, 'Actor'
            ).create_actor()
            self.target_actor = DNN.AC_DNN_TF(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_actor, 'ActorTarget'
            ).create_actor()

            self.critic = DNN.AC_DNN_TF(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_critic, 'Critic'
            ).create_critic()
            self.target_critic = DNN.AC_DNN_TF(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_critic, 'CriticTarget'
            ).create_critic()

            #############################

            self.saver = tf.train.Saver()
            self.checkpoint_file = os.path.join(chkpt_dir, 'ddpg_tf.ckpt')

            #############################

            # Soft Update Operations:

            self.update_target_actor = [self.target_actor.params[i].assign(
                tf.multiply(self.actor.params[i], self.TAU)
                + tf.multiply(self.target_actor.params[i], 1. - self.TAU)
            ) for i in range(len(self.target_actor.params))]

            self.update_target_critic = [self.target_critic.params[i].assign(
                tf.multiply(self.critic.params[i], self.TAU)
                + tf.multiply(self.target_critic.params[i], 1. - self.TAU)
            ) for i in range(len(self.target_critic.params))]

            #############################

            self.sess.run(tf.global_variables_initializer())

            self.update_target_networks_params(first=True)

        def update_target_networks_params(self, first=False):
            if first:
                original_tau = self.TAU
                self.TAU = 1.0
                self.target_actor.ac.sess.run(self.update_target_actor)
                self.target_critic.ac.sess.run(self.update_target_critic)
                self.TAU = original_tau
            else:
                self.target_actor.ac.sess.run(self.update_target_actor)
                self.target_critic.ac.sess.run(self.update_target_critic)

        def choose_action(self, s, noise):
            s = s[np.newaxis, :]

            mu = self.actor.predict(s)[0]
            a = mu + noise  # mu_prime (mu')
            return a

        def learn(self, memory_batch_size, batch_s, batch_a, batch_r, batch_s_, batch_terminal):
            # print('Learning Session')

            batch_target_mu_ = self.target_actor.predict(batch_s_)
            batch_target_q_ = self.target_critic.predict(batch_s_, batch_target_mu_)

            batch_target_q_ = np.reshape(batch_target_q_, memory_batch_size)
            batch_q_target = batch_r + self.GAMMA * batch_target_q_ * batch_terminal
            batch_q_target = np.reshape(batch_q_target, (memory_batch_size, 1))
            self.critic.train(batch_s, batch_a, batch_q_target)

            batch_mu = self.actor.predict(batch_s)
            batch_a_grads = self.critic.get_action_gradients(batch_s, batch_mu)[0]
            self.actor.train(batch_s, batch_a_grads)

            self.update_target_networks_params()

        def load_model_file(self):
            print("...Loading TF checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving TF checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class AC_Torch(object):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, tau, chkpt_dir):

            self.GAMMA = 0.99
            self.TAU = tau

            #############################

            # Networks:

            self.actor = DNN.AC_DNN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_actor, 'Actor', chkpt_dir, is_actor=True
            )
            self.target_actor = DNN.AC_DNN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_actor, 'ActorTarget', chkpt_dir, is_actor=True
            )

            self.critic = DNN.AC_DNN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_critic, 'Critic', chkpt_dir, is_actor=False
            )
            self.target_critic = DNN.AC_DNN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_critic, 'CriticTarget', chkpt_dir, is_actor=False
            )

            #############################

            self.update_target_networks_params(first=True)

        def update_target_networks_params(self, first=False):
            tau = 1.0 if first else self.TAU

            # update_target_actor:
            actor_state_dict = dict(self.actor.named_parameters())
            target_actor_dict = dict(self.target_actor.named_parameters())
            for name in actor_state_dict:
                actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                         (1 - tau) * target_actor_dict[name].clone()
            self.target_actor.load_state_dict(actor_state_dict)

            # update_target_critic:
            critic_state_dict = dict(self.critic.named_parameters())
            target_critic_dict = dict(self.target_critic.named_parameters())
            for name in critic_state_dict:
                critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                          (1 - tau) * target_critic_dict[name].clone()
            self.target_critic.load_state_dict(critic_state_dict)

            # if tau == 1.0
            #     self.verify_copy(tau)

        def verify_copy(self):
            actor_state_dict = dict(self.target_actor.named_parameters())
            print('Verifying Target Actor params have been copied')
            for name, param in self.actor.named_parameters():
                print(name, torch.equal(param, actor_state_dict[name]))

            critic_state_dict = dict(self.target_critic.named_parameters())
            print('Verifying Target Critic params have been copied')
            for name, param in self.critic.named_parameters():
                print(name, torch.equal(param, critic_state_dict[name]))

            input()

        def choose_action(self, s, noise):
            noise = torch.tensor(noise, dtype=torch.float).to(self.actor.device)

            self.actor.eval()
            mu = self.actor.forward(s)
            mu_prime = mu + noise

            return mu_prime.cpu().detach().numpy()

        def learn(self, memory_batch_size, batch_s, batch_a, batch_r, batch_s_, batch_terminal):
            # print('Learning Session')

            batch_r = torch.tensor(batch_r, dtype=torch.float).to(self.critic.device)
            batch_terminal = torch.tensor(batch_terminal).to(self.critic.device)

            self.target_actor.eval()
            batch_target_mu_ = self.target_actor.forward(batch_s_)
            self.target_critic.eval()
            batch_target_q_ = self.target_critic.forward(batch_s_, batch_target_mu_)

            batch_target_q_ = batch_target_q_.view(memory_batch_size)
            batch_q_target = batch_r + self.GAMMA * batch_target_q_ * batch_terminal
            batch_q_target = batch_q_target.view(memory_batch_size, 1).to(self.critic.device)

            self.critic.eval()
            batch_q = self.critic.forward(batch_s, batch_a)

            self.critic.train()
            self.critic.optimizer.zero_grad()
            critic_loss = torch_func.mse_loss(batch_q_target, batch_q)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.actor.eval()
            batch_mu = self.actor.forward(batch_s)
            self.critic.eval()
            batch_q_to_mu = self.critic.forward(batch_s, batch_mu)

            self.actor.train()
            self.actor.optimizer.zero_grad()
            actor_loss = torch.mean(-batch_q_to_mu)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_target_networks_params()

        def load_model_file(self):
            print("...Loading models...")
            self.actor.load_model_file()
            self.target_actor.load_model_file()
            self.critic.load_model_file()
            self.target_critic.load_model_file()

        def save_model_file(self):
            print("...Saving models...")
            self.actor.save_model_file()
            self.target_actor.save_model_file()
            self.critic.save_model_file()
            self.target_critic.save_model_file()


class OUActionNoise(object):

    # https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt  # differential with respect to time - this is a temporally correlated noise
        self.x0 = x0  # initial value
        self.reset()  # resets the temporal correlation

    def reset(self):
        # sets the previous value for the noise
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class Agent(object):

    def __init__(self, custom_env, fc_layers_dims=(400, 300), tau=0.001,  # paper
                 optimizer_type=OPTIMIZER_Adam, lr_actor=0.0001, lr_critic=0.001,  # paper
                 memory_size=None, memory_batch_size=None,
                 device_type=None, lib_type=LIBRARY_TF,
                 base_dir=''):

        self.GAMMA = 0.99  # paper
        self.fc_layers_dims = fc_layers_dims

        self.optimizer_type = optimizer_type
        self.ALPHA = lr_actor
        self.BETA = lr_critic

        self.TAU = tau

        self.noise = OUActionNoise(mu=np.zeros(custom_env.n_actions), sigma=0.2)

        self.memory_size = memory_size if memory_size is not None else custom_env.memory_size
        self.memory_batch_size = memory_batch_size if memory_batch_size is not None else (
            64 if custom_env.input_type == INPUT_TYPE_OBSERVATION_VECTOR else 16  # paper
        )
        self.memory = ReplayBuffer(custom_env, self.memory_size, lib_type, is_discrete_action_space=False)

        # sub_dir = get_file_name(None, self, self.BETA, replay_buffer=True) + '/'
        sub_dir = ''
        self.chkpt_dir = base_dir + sub_dir
        make_sure_dir_exists(self.chkpt_dir)

        if lib_type == LIBRARY_TF:
            self.ac = AC.AC_TF(custom_env, fc_layers_dims,
                               optimizer_type, lr_actor, lr_critic,
                               tau, self.chkpt_dir, device_type)
        else:
            self.ac = AC.AC_Torch(custom_env, fc_layers_dims,
                                  optimizer_type, lr_actor, lr_critic,
                                  tau, self.chkpt_dir)

    def choose_action(self, s, training_mode=False):
        noise = self.noise() if training_mode else 0
        a = self.ac.choose_action(s, noise)
        return a

    def store_transition(self, s, a, r, s_, is_terminal):
        self.memory.store_transition(s, a, r, s_, is_terminal)

    def learn_wrapper(self):
        if self.memory.memory_counter >= self.memory_batch_size:
            self.learn()

    def learn(self):
        # print('Learning Session')

        batch_s, batch_s_, batch_r, batch_terminal, batch_a = \
            self.memory.sample_batch(self.memory_batch_size)

        self.ac.learn(self.memory_batch_size,
                      batch_s, batch_a, batch_r, batch_s_, batch_terminal)

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
            env, 'recordings/DDPG/', force=True,
            video_callable=lambda episode_id: episode_id == 0 or episode_id == (n_episodes - 1)
        )

    print('\n', 'Training Started', '\n')
    train_start_time = datetime.datetime.now()

    starting_ep = learn_episode_index + 1
    for i in range(starting_ep, n_episodes):
        ep_start_time = datetime.datetime.now()

        done = False
        ep_score = 0

        agent.noise.reset()

        observation = env.reset()
        s = custom_env.get_state(observation, None)

        if visualize and i == n_episodes - 1:
            env.render()

        while not done:
            a = agent.choose_action(s, training_mode=True)
            observation_, r, done, info = env.step(a)
            r = custom_env.update_reward(r, done, info)
            s_ = custom_env.get_state(observation_, s)
            ep_score += r
            agent.store_transition(s, a, r, s_, done)
            agent.learn()
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
