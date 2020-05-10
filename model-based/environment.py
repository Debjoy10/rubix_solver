import numpy as np
from collections import Counter
from random import choice, shuffle, randrange
import pycuber as pc
from collections import OrderedDict 
from scipy.stats import mode

class cube:
	def __init__(self, params = {"max_shuffles": 6, "action_space": "small", "definite": False}):

		# Mapping ALL actions to numeric values.
		action_map_large = {'F': 0, 'B': 1, 'U': 2, 'D': 3, 'L': 4, 'R': 5, "F'": 6, "B'": 7, "U'": 8, "D'": 9, "L'": 10, "R'": 11,
			  'F2': 12, 'B2': 13, 'U2': 14, 'D2': 15, 'L2': 16, 'R2': 17, "F2'": 18, "B2'": 19, "U2'": 20, "D2'": 21,
			  "L2'": 22, "R2'": 23}

		# Mapping IMPORTANT actions to numeric values.
		action_map_small = {'F': 0, 'B': 1, 'U': 2, 'D': 3, 'L': 4, 'R': 5, "F'": 6, "B'": 7, "U'": 8, "D'": 9, "L'": 10, "R'": 11}

		# Generating Action space
		if params["action_space"] == "small":
			self.action_map = action_map_small
			self.action_size = 12

		else:
			self.action_map = action_map_large
			self.action_size = 24

		# Mapping Action values back to original actions.
		self.inv_action_map = {v: k for k, v in self.action_map.items()}

		# Mapping all colours to numeric values.
		self.color_map = {'green': 0, 'blue': 1, 'yellow': 2, 'red': 3, 'orange': 4, 'white': 5}

		# One-Hot vector for each colour for state representation
		self.color_list_map = {'green': [1, 0, 0, 0, 0, 0], 'blue': [0, 1, 0, 0, 0, 0], 'yellow': [0, 0, 1, 0, 0, 0],
						  'red': [0, 0, 0, 1, 0, 0], 'orange': [0, 0, 0, 0, 1, 0], 'white': [0, 0, 0, 0, 0, 1]}

		# Other state params
		self.state_size = [6, 3, 3, 6] # --> [6 Face, 3x3 Square, 6 Colour]
		self.flat_state_size = 324
		self.max_reward = 10
		self.max_shuffles = params["max_shuffles"] 

		# Generating the root cube               
		self.state = self.gen_sample(self.max_shuffles, params["definite"])

	# Return a cube with "n" max shuffles
	def gen_sample(self, max_steps, definite = False):
		n_steps = randrange(max_steps+1)
		cube = pc.Cube()
		if(definite):
			n_steps = max_steps

		transformation = [choice(list(self.action_map.keys())) for _ in range(n_steps)]
		my_formula = pc.Formula(transformation)
		cube(my_formula)
		return cube

	# Return a cube sequence from shuffled state to goal state
	def gen_sequence(self, max_steps):
		n_steps = randrange(max_steps+1)
		cube = pc.Cube()

		transformation = [choice(list(self.action_map.keys())) for _ in range(n_steps)]
		my_formula = pc.Formula(transformation)
		cube(my_formula)
		
		my_formula.reverse()
		cube_seq = []

		for i, s in enumerate(my_formula):
			cube_seq.append(cube.copy())
			cube(s.name)
		cube_seq.append(cube.copy())

		return cube_seq, my_formula 

	# Cube to 4D array
	def to_4d_array(self):
		cube4d = np.zeros(self.state_size)
		sides = [self.state.F, self.state.B, self.state.U, self.state.D, self.state.L, self.state.R]

		for face in range(self.state_size[0]):
			for i in range(self.state_size[1]):
				for j in range(self.state_size[2]):
					cube4d[face][i][j] = self.color_list_map[sides[face][i][j].colour]

		return cube4d

	# Cube to 2D array (All Squares x Colours --> 54 x 6)
	def to_2d_array(self):
		sides = [self.state.F, self.state.B, self.state.U, self.state.D, self.state.L, self.state.R]

		cube2d = []
		for face in sides:
			for i in range(self.state_size[1]):
				for j in range(self.state_size[2]):
					cube2d.append(self.color_list_map[face[i][j].colour])
		
		return np.array(cube2d)

	# Cube to 1D array (Flatten --> 324 x 1)
	def flatten(self, state):
		sides = [state.F, state.B, state.U, state.D, state.L, state.R]

		flat = []
		for face in sides:
			for i in range(self.state_size[1]):
				for j in range(self.state_size[2]):
					flat.extend(self.color_list_map[face[i][j].colour])
		
		return np.array(flat)

	# Utility Function For Percentage
	def order(self, data):
		if len(data) <= 1:
			return 0
		counts = Counter()
		for d in data:
			counts[d] += 1
		probs = [float(c) / len(data) for c in counts.values()]
		return max(probs)

	#Colour-Flatten: Utility Function For Percentage
	def flatten_color(self, cube):
		sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
		flat = []
		for x in sides:
			for i in range(3):
				for j in range(3):
					flat.append(x[i][j].colour)
		return flat

	# Percentage solved cube
	def perc_solved_cube(self):
		flat = self.flatten_color(self.state)
		perc_side = [self.order(flat[i:(i + 9)]) for i in range(0, 9 * 6, 9)]
		return np.mean(perc_side)

	# Returns reward for action
	def get_reward(self):
		# Returns reward, done
		# Reward Solved => 100
		#        Unsolved => -1
		if self.perc_solved_cube() > 0.9:
			return self.max_reward, True
		else: return -1, False

	# Takes a Step Online/Offline
	def step(self, action):
		a = self.inv_action_map[action]
		self.state(a)
		reward, done = self.get_reward()
		info = None
		return self.state, reward, done, info
