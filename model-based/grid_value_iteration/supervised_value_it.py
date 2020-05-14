import sys
sys.path.insert(0, '..')

from random import choice, shuffle, randrange
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from tqdm import tqdm
from environment import cube
import numpy as np
from datetime import datetime
import json


# Prepare Random samples for Value Function Iteration
def generate_samples():
	MAX_STEPS = 20
	params =  {"max_shuffles": 0, "action_space": "small", "definite": False}
	env = cube(params)

	curr_value = env.max_reward
	seq = [env.flatten(env.state.copy())]
	val = [curr_value]

	for i in range(MAX_STEPS):
		action = randrange(env.action_size)
		state, reward, done, info = env.step(action)
		curr_value += reward

		seq.append(env.flatten(state = state.copy()))
		val.append(curr_value)

	return seq, val

# Prepare training dataset for Value Function Iteration
def generate_dataset(data_size):
	try:
		dataset = np.load('dataset.npy',allow_pickle='TRUE').item()

	except:
		state_sequences = []
		value_sequences = []

		print("Building Dataset")
		for i in tqdm(range(data_size)):
			seq, val = generate_samples()
			state_sequences.extend(seq)
			value_sequences.extend(val)

		dataset = {"state_sequences": np.array(state_sequences), "value_sequences": np.array(value_sequences)}
		np.save('dataset.npy', dataset)
	
	return dataset

# Define Model in tf.keralocas
def build_model(batch_size = 32):
  model = tf.keras.models.Sequential([
	  tf.keras.layers.Flatten(input_shape=(324,)),
	  tf.keras.layers.Dense(128, activation='relu'),
	  tf.keras.layers.Dropout(0.2),
	  tf.keras.layers.Dense(32, activation='relu'),
	  tf.keras.layers.Dropout(0.2),
	  tf.keras.layers.Dense(1)
	])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
				optimizer=optimizer,
				metrics=['mae', 'mse'])
  return model

def make_dataset(data_size = 10000):
	npds = generate_dataset(data_size)

	train_examples = npds["state_sequences"]
	train_labels = npds["value_sequences"]
	train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
	return train_dataset

def train(SHUFFLE_BUFFER_SIZE = 100, BATCH_SIZE = 32):
	model = build_model()
	model.summary()

	train_dataset = make_dataset()
	train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

	logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
	model.fit(train_dataset, epochs=500)
	save(model)
	return model

def save(model):
	model.save('./model/')

def load():
	loaded_model = tf.keras.models.load_model('./model/')
	return loaded_model

model = train()