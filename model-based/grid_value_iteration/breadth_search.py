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
	loaded_model = tf.keras.models.load_model('./model/')
	return loaded_model

def is_end(cube):
	flat = env.flatten_color(cube)
	perc_side = [env.order(flat[i:(i + 9)]) for i in range(0, 9 * 6, 9)]
	if np.mean(perc_side) == 1:
		return True
	return False

def breadth_find_best(state, depth, action_seq):

	if is_end(state):
		return 1000, action_seq

	if depth == 0:
		st = np.float64(np.array(env.flatten(state))).reshape(1, 324)
		v = model.predict(st)
		return v, action_seq
	
	all_vals = np.zeros(env.action_size)
	for action in range(0, env.action_size):
		a = env.inv_action_map[action]
		cs = state.copy()
		cs(a)

		val, action_seq = breadth_find_best(cs, depth-1, action_seq)
		all_vals[action] = val

	best_value = np.max(all_vals)
	action_seq.append(np.argmax(all_vals))

	return best_value, action_seq 


model = load()
params =  {"max_shuffles": 4, "action_space": "small", "definite": True}
env = cube(params)
done = False
total_reward = 0
stepnum = 0

while not done:
	maxv = 0
	imax = 0
	if is_end(env.state):
		break

	stepnum += 1
	print([env.state])

	v, aseq = breadth_find_best(env.state.copy(), 2, [])
	imax = aseq[-1]	

	print("Step --> {}".format(stepnum))
	newstate, reward, done, info = env.step(imax)
	total_reward += reward

print([env.state])
print("Episode Finished")
print("Total Reward = {}".format(total_reward))