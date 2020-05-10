import numpy as np
from collections import Counter
from random import choice, shuffle, randrange
import pycuber as pc
from collections import OrderedDict 

from environment import cube

mycube = cube(5)
print([mycube.state])

# print(mycube.to_2d_array().shape)
# print(mycube.to_4d_array().shape)
# print(mycube.flatten().shape)

# print(mycube.perc_solved_cube())
# print(mycube.step(5))
# print([mycube.state])

print(mycube.gen_sequence(7))