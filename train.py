import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import action_map_small, gen_sequence, get_all_possible_actions_cube_small, chunker, flatten_1d_b

gamma = 0.99

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

class agent():
	def __init__(self, lr, s_size,a_size,h_size):
		#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
		self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
		hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
		self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.tanh,biases_initializer=None)
		self.chosen_action = tf.argmax(self.output,1)
		# Also calculate the value of future state
		# self.value = slim.fully_connected(hidden,1,activation_fn=tf.nn.tanh,biases_initializer=tf.contrib.layers.xavier_initializer)

		#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
		#to compute the loss, and use it to update the network.
		self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
		self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
		
		self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
		self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

		self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
		
		tvars = tf.trainable_variables()
		self.gradient_holders = []
		for idx,var in enumerate(tvars):
			placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
			self.gradient_holders.append(placeholder)
		
		self.gradients = tf.gradients(self.loss,tvars)
		
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


tf.reset_default_graph() #Clear the Tensorflow graph.

# myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.
myAgent = agent(lr=1e-2,s_size=324,a_size=12,h_size=64)
# state_size
# action_size
# hidden_layer_size (guessing)

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0
	total_reward = []
	total_length = []
		
	gradBuffer = sess.run(tf.trainable_variables())
	for ix,grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0
	

	n_samples = 1
	for i in tqdm(range(total_episodes)):
		cubes = []
		distance_to_solved = []
		
		for j in tqdm(range(n_samples)):
			cube_list, dist_list = gen_sequence(25)
			cubes.extend(cube_list)
			distance_to_solved.extend(dist_list)

		# 2500 cubes generated

		print("\n ---Samples generated")
		# gen_sequence(k) returns k cubes indexed i = [1, 2, ..., k] i moves away from final

		cube_next_reward = []
		flat_next_states = []
		cube_flat = []

		for index in tqdm(range(len(cubes))):
			c = cubes[index]
			d = distance_to_solved[index]
			print([c])
			flat_cubes, rewards = get_all_possible_actions_cube_small(c)
			cube_next_reward.append(rewards)
			flat_next_states.extend(flat_cubes)
			cube_flat.append(flatten_1d_b(c))

		print("\n ---All actions noted")
		cube_target_policy = []
		cube_target_value = []

		# Probabilistically pick an action given our network outputs.
		next_state_value = sess.run(myAgent.output,feed_dict={myAgent.state_in:np.array(flat_next_states)})
		a = [np.argmax(act) for act in next_state_value]
		print("predicted actions: "+str(a))

		for c, rewards, values in tqdm(zip(cubes, cube_next_reward, next_state_value)):
			r_plus_v = np.array(rewards) + gamma*np.array(values)
			target_v = np.max(r_plus_v)
			target_p = np.argmax(r_plus_v)
			cube_target_value.append(target_v)
			cube_target_policy.append(target_p)

		# Normalizing Values
		cube_target_value = (cube_target_value-np.mean(cube_target_value))/(np.std(cube_target_value)+0.001)
		

		feed_dict={myAgent.reward_holder:cube_target_value,
				myAgent.action_holder:cube_target_policy,myAgent.state_in:np.array(cube_flat)}
		grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
		for idx,grad in enumerate(grads):
			gradBuffer[idx] += grad

		# if i % update_frequency == 0 and i != 0:
		if True:
			print("training")
			feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
			_ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
			for ix,grad in enumerate(gradBuffer):
				gradBuffer[ix] = grad * 0

		if i%5 == 0:
			print("yay")

		#Update our running tally of scores.
		if i % 100 == 0:
			print(np.mean(total_reward[-100:]))
