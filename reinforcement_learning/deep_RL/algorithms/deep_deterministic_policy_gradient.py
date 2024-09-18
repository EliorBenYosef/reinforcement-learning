"""
https://arxiv.org/pdf/1509.02971.pdf
"""

import datetime
import os
import numpy as np
from gym import wrappers

import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.ops.variables import trainable_variables
# import tensorflow.keras.backend as keras_backend
import torch
import torch.nn.functional as torch_func
import torch.nn.init as torch_init

from reinforcement_learning.utils.utils import print_training_progress, pickle_save, make_sure_dir_exists
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, OPTIMIZER_Adam,\
    INPUT_TYPE_OBSERVATION_VECTOR
from reinforcement_learning.deep_RL.utils.utils import OUActionNoise
from reinforcement_learning.deep_RL.utils.saver_loader import load_training_data, save_training_data
from reinforcement_learning.deep_RL.utils.optimizers import tf_get_optimizer, torch_get_optimizer
from reinforcement_learning.deep_RL.utils.devices import tf_get_session_according_to_device, \
    torch_get_device_according_to_device_type
from reinforcement_learning.deep_RL.utils.replay_buffer import ReplayBuffer


class NN:

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
            return NN.NN_TensorFlow.Actor(self)

        def create_critic(self):
            return NN.NN_TensorFlow.Critic(self)

        class Actor(object):

            def __init__(self, nn):
                self.nn = nn

                self.build_network()

                self.params = trainable_variables(scope=self.nn.name)

                # actor loss: mean(-batch_q_of_a_pred)
                self.mu_grads = tf.gradients(self.mu, self.params, -self.a_grads)
                self.normalized_mu_grads = list(map(lambda x: tf.div(x, self.nn.memory_batch_size), self.mu_grads))

                optimizer = tf_get_optimizer(self.nn.optimizer_type, self.nn.lr)
                self.optimize = optimizer.apply_gradients(zip(self.normalized_mu_grads, self.params))  # train_op

            def build_network(self):
                with tf.compat.v1.variable_scope(self.nn.name):
                    self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.nn.input_dims], name='s')
                    self.a_grads = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nn.n_actions], name='a_grads')

                    f1 = 1. / np.sqrt(self.nn.fc_layers_dims[0])
                    x = tf.compat.v1.layers.dense(self.s, units=self.nn.fc_layers_dims[0],
                                        kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                                        bias_initializer=tf.random_uniform_initializer(-f1, f1))
                    x = tf.compat.v1.layers.batch_normalization(x)
                    x = tf.nn.relu(x)

                    f2 = 1. / np.sqrt(self.nn.fc_layers_dims[1])
                    x = tf.compat.v1.layers.dense(x, units=self.nn.fc_layers_dims[1],
                                        kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                                        bias_initializer=tf.random_uniform_initializer(-f2, f2))
                    x = tf.compat.v1.layers.batch_normalization(x)
                    x = tf.nn.relu(x)

                    f3 = 0.003
                    mu = tf.compat.v1.layers.dense(x, units=self.nn.n_actions, activation='tanh',
                                         kernel_initializer=tf.random_uniform_initializer(-f3, f3),
                                         bias_initializer=tf.random_uniform_initializer(-f3, f3))
                    self.mu = tf.multiply(mu, self.nn.action_boundary)  # an ndarray of ndarrays

            def train(self, s, a_grads):
                # print('Training Started')
                self.nn.sess.run(self.optimize,
                                 feed_dict={self.s: s,
                                            self.a_grads: a_grads})
                # print('Training Finished')

            def predict(self, s):
                return self.nn.sess.run(self.mu,
                                        feed_dict={self.s: s})

        class Critic(object):

            def __init__(self, nn):
                self.nn = nn

                self.build_network()

                self.params = trainable_variables(scope=self.nn.name)

                optimizer = tf_get_optimizer(self.nn.optimizer_type, self.nn.lr)
                self.optimize = optimizer.minimize(self.loss)  # train_op

                self.a_grads = tf.gradients(self.q, self.a)  # a list containing an ndarray of ndarrays

            def build_network(self):
                with tf.compat.v1.variable_scope(self.nn.name):
                    self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.nn.input_dims], name='s')
                    self.a = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nn.n_actions], name='a')
                    self.q_target = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='q_target')

                    f1 = 1. / np.sqrt(self.nn.fc_layers_dims[0])
                    x = tf.compat.v1.layers.dense(self.s, units=self.nn.fc_layers_dims[0],
                                        kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                                        bias_initializer=tf.random_uniform_initializer(-f1, f1))
                    x = tf.compat.v1.layers.batch_normalization(x)
                    x = tf.nn.relu(x)

                    f2 = 1. / np.sqrt(self.nn.fc_layers_dims[1])
                    x = tf.compat.v1.layers.dense(x, units=self.nn.fc_layers_dims[1],
                                        kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                                        bias_initializer=tf.random_uniform_initializer(-f2, f2))
                    x = tf.compat.v1.layers.batch_normalization(x)

                    action_in_ac = tf.compat.v1.layers.dense(self.a, units=self.nn.fc_layers_dims[1], activation='relu')

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
                self.nn.sess.run(self.optimize,
                                 feed_dict={self.s: s,
                                            self.a: a,
                                            self.q_target: q_target})
                # print('Training Finished')

            def predict(self, s, a):
                return self.nn.sess.run(self.q,
                                        feed_dict={self.s: s,
                                                   self.a: a})

            def get_action_gradients(self, inputs, actions):
                return self.nn.sess.run(self.a_grads,
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

        def load_model_file(self):
            self.load_state_dict(torch.load(self.model_file))

        def save_model_file(self):
            torch.save(self.state_dict(), self.model_file)

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


class AC(object):

    class AC_TF(object):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, tau, chkpt_dir, device_map):

            self.GAMMA = 0.99
            self.TAU = tau

            self.sess = tf_get_session_according_to_device(device_map)

            #############################

            # Networks:

            self.actor = NN.NN_TensorFlow(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_actor, 'Actor'
            ).create_actor()
            self.target_actor = NN.NN_TensorFlow(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_actor, 'ActorTarget'
            ).create_actor()

            self.critic = NN.NN_TensorFlow(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_critic, 'Critic'
            ).create_critic()
            self.target_critic = NN.NN_TensorFlow(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_critic, 'CriticTarget'
            ).create_critic()

            #############################

            self.saver = tf.compat.v1.train.Saver()
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

            self.sess.run(tf.compat.v1.global_variables_initializer())

            self.update_target_networks_params(first=True)

        def update_target_networks_params(self, first=False):
            if first:
                original_tau = self.TAU
                self.TAU = 1.0
                self.target_actor.nn.sess.run(self.update_target_actor)
                self.target_critic.nn.sess.run(self.update_target_critic)
                self.TAU = original_tau
            else:
                self.target_actor.nn.sess.run(self.update_target_actor)
                self.target_critic.nn.sess.run(self.update_target_critic)

        def choose_action(self, s, noise):
            s = s[np.newaxis, :]

            mu = self.actor.predict(s)[0]
            a = mu + noise  # mu_prime (mu')
            return a

        def learn(self, memory_batch_size, batch_s, batch_a_true, batch_r, batch_s_, batch_terminal):
            # print('Learning Session')

            batch_a_ = self.target_actor.predict(batch_s_)
            batch_q_ = self.target_critic.predict(batch_s_, batch_a_)
            batch_q_ = np.reshape(batch_q_, memory_batch_size)

            batch_q_target = batch_r + self.GAMMA * batch_q_ * batch_terminal
            batch_q_target = np.reshape(batch_q_target, (memory_batch_size, 1))

            self.critic.train(batch_s, batch_a_true, batch_q_target)  # batch_s, batch_a_true --> batch_q_of_a_true

            batch_a_pred = self.actor.predict(batch_s)
            batch_a_grads = self.critic.get_action_gradients(batch_s, batch_a_pred)[0]  # batch_s, batch_a_pred --> batch_q_of_a_pred
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

            self.actor = NN.NN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_actor, 'Actor', chkpt_dir, is_actor=True
            )
            self.target_actor = NN.NN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_actor, 'ActorTarget', chkpt_dir, is_actor=True
            )

            self.critic = NN.NN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_critic, 'Critic', chkpt_dir, is_actor=False
            )
            self.target_critic = NN.NN_Torch(
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
            noise = torch.tensor(noise, dtype=torch.float32).to(self.actor.device)

            self.actor.eval()
            mu = self.actor.forward(s)
            mu_prime = mu + noise

            return mu_prime.cpu().detach().numpy()

        def learn(self, memory_batch_size, batch_s, batch_a_true, batch_r, batch_s_, batch_terminal):
            # print('Learning Session')

            batch_r = torch.tensor(batch_r, dtype=torch.float32).to(self.critic.device)
            batch_terminal = torch.tensor(batch_terminal).to(self.critic.device)

            self.target_actor.eval()
            batch_a_ = self.target_actor.forward(batch_s_)
            self.target_critic.eval()
            batch_q_ = self.target_critic.forward(batch_s_, batch_a_)
            batch_q_ = batch_q_.view(memory_batch_size)

            batch_q_target = batch_r + self.GAMMA * batch_q_ * batch_terminal
            batch_q_target = batch_q_target.view(memory_batch_size, 1).to(self.critic.device)

            self.critic.eval()
            batch_q_of_a_true = self.critic.forward(batch_s, batch_a_true)
            self.critic.train()
            self.critic.optimizer.zero_grad()
            critic_loss = torch_func.mse_loss(batch_q_target, batch_q_of_a_true)
            # l1_crit = torch.nn.Loss(size_average=False)
            # reg_loss = 0
            # for param in model.parameters():
            #     reg_loss += l1_crit(param)
            # factor = 0.0005
            # loss += factor * reg_loss
            critic_loss.backward()
            self.critic.optimizer.step()

            self.actor.eval()
            batch_a_pred = self.actor.forward(batch_s)
            self.critic.eval()
            batch_q_of_a_pred = self.critic.forward(batch_s, batch_a_pred)

            self.actor.train()
            self.actor.optimizer.zero_grad()
            actor_loss = torch.mean(-batch_q_of_a_pred)
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
            reset_default_graph()
            self.ac = AC.AC_TF(custom_env, fc_layers_dims,
                               optimizer_type, lr_actor, lr_critic,
                               tau, self.chkpt_dir, device_type)
        # elif lib_type == LIBRARY_KERAS:
        #     keras_backend.clear_session()
        #     pass
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

        batch_s, batch_s_, batch_r, batch_terminal, batch_a_continuous = \
            self.memory.sample_batch(self.memory_batch_size)

        self.ac.learn(self.memory_batch_size,
                      batch_s, batch_a_continuous, batch_r, batch_s_, batch_terminal)

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
            agent.learn_wrapper()
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
