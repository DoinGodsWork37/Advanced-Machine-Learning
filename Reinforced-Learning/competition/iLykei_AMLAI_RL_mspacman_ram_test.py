
# coding: utf-8

# # iLykei Lecture Series
# 
# # Advanced Machine Learning and Artificial Intelligence (MScA 32017)
# 
# # Pac-Man Competition for Human-Machine Teams 
# 
# ### Y.Balasanov, M. Tselishchev, &copy; iLykei 2018
# 
# ## Preparation

# In[1]:


import random
import numpy as np
import gym


# Load trained model (which was previously saved by `model.save()`-method) for online network:

# In[2]:


from keras.models import load_model
online_network = load_model('pacman1.h5', compile=False) 


# Define $\varepsilon$-greedy strategy (using small $\varepsilon$):

# In[3]:


def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randrange(n_outputs)  # random action
    else:
        return np.argmax(q_values)          # q-optimal action


# ## Testing model
# 
# Define a function to evalutate the trained network. 
# Note that we still using $\varepsilon$-greedy strategy here to prevent an agent from getting stuck. 
# `test_dqn` returns a list with scores for specific number of games.

# In[4]:


def test_dqn(n_games, model, nb_actions=9, skip_start=90, eps=0.05, render=False, sleep_time=0.01):
    env = gym.make("MsPacman-ram-v0")
    scores = []
    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)
            obs, reward, done, info = env.step(0)
            score += reward
        while not done:
            state = obs
            q_values = model.predict(np.array([state]))[0]
            action = epsilon_greedy(q_values, eps, nb_actions)
            obs, reward, done, info = env.step(action)
            score += reward
            if render:
                env.render()
                time.sleep(sleep_time)
                if done:
                    time.sleep(1)
        scores.append(score)
        print('{}/{}: {}'.format(i+1, n_games, score))
        env.close()
    return scores


# ### Collecting scores
# 
# Run 100 games without rendering and collect necessary statistics for final score.

# In[5]:


ngames = 100
eps = 0.05
render = False

scores = test_dqn(ngames, online_network, eps=eps, render=render)

print('\nMean score: ', np.mean(scores))
print('\nMax score: ', np.max(scores))
print('\nPercentiles:')
print([ np.percentile(scores, p) for p in [0, 25, 50, 75, 100] ])


# ### Rendering
# 
# Play 3 more times with rendering

# In[6]:


import time
ngames = 3
eps = 0.05
render = True

scores = test_dqn(ngames, online_network, eps=eps, render=render)

print('\nMean score: ', np.mean(scores))
print('\nMax score: ', np.max(scores))
print('\nPercentiles:')
print([ np.percentile(scores, p) for p in [0, 25, 50, 75, 100] ])

