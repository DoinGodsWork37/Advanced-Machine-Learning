import gym
import pickle

env = gym.make("MsPacman-ram-v0")

def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(env, "ms_pacman.obj")
