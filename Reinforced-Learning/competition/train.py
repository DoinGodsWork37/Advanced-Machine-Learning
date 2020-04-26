#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import gc
import time
import numpy as np
import os.path
from keras.models import Sequential, clone_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, TensorBoard
import keras.backend as K
import pyglet
from collections import deque
import json
import pickle

import random
import numpy as np
import gym


# In[2]:


def create_dqn_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Dense(units=32, input_shape=input_shape, activation="relu"))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(nb_actions, activation="linear"))
    return model


# In[4]:


def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randrange(n_outputs)  # random action
    else:
        return np.argmax(q_values)          # q-optimal action


# In[5]:


n_steps = 1_000_000 # number of times
warmup = 50_000 # first iterations after random initiation before training starts
training_interval = 4 # number of steps after which dqn is retrained
copy_steps = 10_000 # number of steps after which weights of
                   # online network copied into target network
gamma = 0.99 # discount rate
batch_size = 256 # size of batch from replay memory
eps_max = 1.0 # parameters of decaying sequence of eps
eps_min = 0.1
eps_decay_steps = 500_000

replay_memory_maxlen = 20_000_000 #atleast training interval * number of steps
replay_memory = deque([], maxlen=replay_memory_maxlen)


# weights_path = 'weights'
# if not os.path.exists(weights_path):
#     os.makedirs(weights_path)
# weights_path += '/pacman1.hdf5'
# 

# In[6]:

env = gym.make("MsPacman-ram-v0")

def train_dqn_net():
    input_shape = env.observation_space.shape
    nb_actions = env.action_space.n
    print('input_shape: ', input_shape)
    print('nb_actions: ', nb_actions)

    online_network = create_dqn_model(input_shape, nb_actions)
    online_network.compile(optimizer=Adam(), loss='mse')
    target_network = clone_model(online_network)
    target_network.set_weights(online_network.get_weights())

    step = 0
    iteration = 0
    done = True

    while step < n_steps:
        if done:
            obs = env.reset()
        iteration += 1
        q_values = online_network.predict(np.array([obs]))[0]
        epsilon = max(eps_min,
                      eps_max - (eps_max - eps_min) * step / eps_decay_steps)
        action = epsilon_greedy(q_values, epsilon, nb_actions)
        next_obs, reward, done, info = env.step(action)
        replay_memory.append((obs, action, reward, next_obs, done))
        obs = next_obs

        if iteration >= warmup and iteration % training_interval == 0:
            step += 1
            minibatch = random.sample(replay_memory, batch_size)
            replay_state = np.array([x[0] for x in minibatch])
            replay_action = np.array([x[1] for x in minibatch])
            replay_rewards = np.array([x[2] for x in minibatch])
            replay_next_state = np.array([x[3] for x in minibatch])
            replay_done = np.array([x[4] for x in minibatch], dtype=int)
            target_for_action = replay_rewards + (1 - replay_done) * gamma * \
                                np.amax(
                                    target_network.predict(replay_next_state),
                                    axis=1)
            target = online_network.predict(
                replay_state)  # targets coincide with predictions ...
            target[np.arange(
                batch_size), replay_action] = target_for_action  # ...except for targets with actions from replay
            online_network.fit(replay_state, target, epochs=step, verbose=1,
                               initial_epoch=step - 1)
            if step % copy_steps == 0:
                target_network.set_weights(online_network.get_weights())
    online_network.save("pacman1.h5")


# In[ ]:


train_dqn_net()

