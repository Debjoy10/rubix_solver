# RubiX Solver
Autonomous Rubic's cube solver using Reinforcement Learning and Tree search.
###### (Work In Progress - Latest Version can be found in model-based with usage instructions given)        

Approaches developed Up till now-
1. **naiveQ** algorithm is based on a model-free Q-learning algorithm. Solve states at maximum 1 step away from solved state(1 scramble ONLY).  
More detailed README provided.
2. **model-based** (Under Development) aims at using an AlphaZero based approach for solving rubik's cube. *Current Progress*- Value function estimation in Supervised setting for states upto 10 scrambles. Trained agent plays successfully upto 7 samples.  
See Learned agent play for max 5 scrambles - Run `python3 test.py` in model-based/grid_value_iteration folder.
