import sys
sys.path.insert(0, '..')
from copy import deepcopy
from copy import copy

from random import choice, shuffle, randrange
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from tqdm import tqdm
from environment import cube
import numpy as np

def load():
	loaded_model = tf.keras.models.load_model('/tmp/model')
	return loaded_model

def find_next_v(state, action, env):
	a = env.inv_action_map[action]
	cs = state.copy()
	cs(a)
	st = np.float64(np.array(env.flatten(cs))).reshape(1, 324)
	v = model.predict(st)
	return v
	

model = load()
params =  {"max_shuffles": 6, "action_space": "small", "definite": True}
env = cube(params)
done = False
total_reward = 0
stepnum = 0

while not done:
	maxv = 0
	stepnum += 1

	print([env.state])

	for i in range(12):
		v = find_next_v(env.state.copy(), i, env)
		if v > maxv:
			maxv = v
			imax = i

	print("Step --> {}".format(stepnum))
	newstate, reward, done, info = env.step(imax)
	total_reward += reward


print([env.state])
print("Episode Finished")
print("Total Reward = {}".format(total_reward))
