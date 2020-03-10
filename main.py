import os

from prng import prng, LEHMER_0, LEHMER_1, LEHMER_2, LEHMER_3, LEHMER_4, LEHMER_5, MRG, ICG, MIXED_CG, EICG
from Solver import Solver


def main():
    vrp_set_path = './Vrp-Set-E'
    problem_path = os.path.join(vrp_set_path, 'E-n30-k3.vrp')

    solver = Solver(problem_path)

    cost, route = solver.binary_cws_mcs(n=50, distributed=True)
    print(cost,route)

main()
