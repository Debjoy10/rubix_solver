# Trying out the pycuber functions
import utils
import pycuber as pc

mycube = pc.Cube()

# mycube("R U R' U'")
my_formula = pc.Formula("R U R' U' R' F R2 U' R' U' R U R' F'")
my_formula = pc.Formula("")
print(my_formula)

print(mycube(my_formula))

# c, dc = utils.gen_sequence(10)
# print(c)
# for cube in c:
# 	print(cube)
# 	print(utils.perc_solved_cube(cube))