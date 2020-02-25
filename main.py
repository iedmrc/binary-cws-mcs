import os

from prng import LEHMER_0, LEHMER_2, LEHMER_5, ICG, MIXED_CG
from Solver import Solver

generators = [LEHMER_0, LEHMER_2, LEHMER_5, ICG, MIXED_CG]

def main():
    vrp_set_path = './Vrp-Set-E'
    problem_path = os.path.join(vrp_set_path, 'E-n101-k14.vrp')

    for generator in generators:
        solver = Solver(problem_path, generator)
        cost, route_list = solver.binary_cws()
        print("cost: ",cost)
        print("route: ", route_list)
        print("------------------------------")


main()
