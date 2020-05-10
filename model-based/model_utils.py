from collections import Counter
from random import choice, shuffle, randrange
import numpy as np
import pycuber as pc
from collections import OrderedDict 

# Mapping all actions to numeric values.
action_map = {'F': 0, 'B': 1, 'U': 2, 'D': 3, 'L': 4, 'R': 5, "F'": 6, "B'": 7, "U'": 8, "D'": 9, "L'": 10, "R'": 11,
              'F2': 12, 'B2': 13, 'U2': 14, 'D2': 15, 'L2': 16, 'R2': 17, "F2'": 18, "B2'": 19, "U2'": 20, "D2'": 21,
              "L2'": 22, "R2'": 23}

# Mapping Action values back to original actions.
inv_action_map = {v: k for k, v in action_map.items()}

# Mapping all colours to numeric values.
color_map = {'green': 0, 'blue': 1, 'yellow': 2, 'red': 3, 'orange': 4, 'white': 5}

# One-Hot vector for each colour for state representation
color_list_map = {'green': [1, 0, 0, 0, 0, 0], 'blue': [0, 1, 0, 0, 0, 0], 'yellow': [0, 0, 1, 0, 0, 0],
                  'red': [0, 0, 0, 1, 0, 0], 'orange': [0, 0, 0, 0, 1, 0], 'white': [0, 0, 0, 0, 0, 1]}


# Return a cube with "n" max shuffles
def gen_sample(max_steps=6):
	n_steps = randrange(max_steps+1)
	cube = pc.Cube()

	transformation = [choice(list(action_map.keys())) for _ in range(n_steps)]
	my_formula = pc.Formula(transformation)
	cube(my_formula)
	return cube

cube3d = np.zeros([6, 9, 6])
print(cube3d)


