import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import action_map_small, gen_sequence, get_all_possible_actions_cube_small, chunker, flatten_1d_b, perc_solved_cube
import pycuber as pc
from random import choice, shuffle
import time

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
saver = tf.train.Saver()

with tf.Session() as sess:
    
    cube = pc.Cube()
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    # saver.restore(sess, "./best_model/model.ckpt")

    saa = int(input("Shuffle : ") )

    cube = pc.Cube()
    transformation = [choice(list(action_map_small.keys())) for z in range(saa)]
    my_formula = pc.Formula(transformation)
    cube(my_formula)
    

    print([cube])
    print("********************************************************")
    flat_cube = np.array(flatten_1d_b(cube)).reshape([1, 324])
    Qs = sess.run(myAgent.output, feed_dict = {myAgent.state_in: flat_cube})
    print("Q values:")
    print(Qs)
