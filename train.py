import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import action_map_small, gen_sequence, get_all_possible_actions_cube_small, chunker, flatten_1d_b, gen_sequence_plus_1

gamma = 0.99

action_list = ["R", "L","D","U","B","F","R'", "L'","D'","U'","B'","F'"]

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

class agent():
	def __init__(self, lr, s_size,a_size,h1_size,h2_size):
		#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
		self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32, name = "input_")
		self.actions_ = tf.placeholder(tf.float32, [None, a_size], name="actions_")
			
		# Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
		self.target_Q = tf.placeholder(tf.float32, [None], name="target")

		hidden1 = slim.fully_connected(self.state_in,h1_size,biases_initializer=None,activation_fn=tf.nn.tanh)
		hidden2 = slim.fully_connected(hidden1,h2_size,biases_initializer=None,activation_fn=tf.nn.tanh)
		self.output = tf.layers.dense(inputs = hidden2, kernel_initializer=tf.contrib.layers.xavier_initializer(), units = a_size, activation=None)
		
		self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)       
		# The loss is the difference between our predicted Q_values and the Q_target
		# Sum(Qtarget - Q)^2
		self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
		self.optimizer = tf.train.RMSPropOptimizer(lr).minimize(self.loss)


tf.reset_default_graph() #Clear the Tensorflow graph.

# myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.
myAgent = agent(lr=1e-2,s_size=324,a_size=12,h1_size=64, h2_size=32)
# state_size
# action_size
# hidden_layer_sizes (guessing)

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999

init = tf.global_variables_initializer()

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("./summaries/tensorboard/dqn/1")
# writer = tf.summary.FileWriter("/home/debjoy/A_projects/cube_solver/rubix_solver")

## Losses
tf.summary.scalar("Loss", myAgent.loss)
write_op = tf.summary.merge_all()

saver = tf.train.Saver()

# Launch the tensorflow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0
	total_reward = []
	total_length = []
		
	n_samples = 1
	for i in tqdm(range(total_episodes)):
		cubes = []
		distance_to_solved = []
		
		for j in tqdm(range(n_samples)):
			cube_list, dist_list = gen_sequence_plus_1(25)
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
			flat_cubes, rewards, acts = get_all_possible_actions_cube_small(c)
			cube_next_reward.append(rewards)
			flat_next_states.extend(flat_cubes)
			cube_flat.append(flatten_1d_b(c))

		print("\n ---All actions noted")
		cube_target_policy = []
		cube_target_value = []
		action_value = []

		# Probabilistically pick an action given our network outputs.
		next_state_value = sess.run(myAgent.output,feed_dict={myAgent.state_in:np.array(flat_next_states)})
		action = [np.argmax(act) for act in next_state_value]

		for c, rewards, values in tqdm(zip(cubes, cube_next_reward, next_state_value)):
			r_plus_v = np.array(rewards) + gamma*np.array(values)
			target_v = np.max(r_plus_v)
			target_p = np.argmax(r_plus_v) 

			cube_target_value.append(target_v)
			cube_target_policy.append(target_p)

			actval = np.zeros(12)
			actval[target_p] = 1
			action_value.append(actval)

		loss, _ = sess.run([myAgent.loss, myAgent.optimizer], feed_dict={myAgent.state_in: np.array(cube_flat), myAgent.target_Q: np.array(cube_target_value), myAgent.actions_: np.array(action_value)})

		# Write TF Summaries
		summary = sess.run(write_op, feed_dict={myAgent.state_in: np.array(cube_flat), myAgent.target_Q: np.array(cube_target_value), myAgent.actions_: np.array(action_value)})
		writer.add_summary(summary, i)
		writer.flush()
		print("LOSS: "+str(loss))

		# Save model every 5 episodes
		if i % 5 == 0:
			save_path = saver.save(sess, "./models/model.ckpt")
			print("Model Saved")
		
