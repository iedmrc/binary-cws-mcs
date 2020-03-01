import os

from prng import prng, LEHMER_0, LEHMER_1, LEHMER_2, LEHMER_3, LEHMER_4, LEHMER_5, MRG, ICG, MIXED_CG, EICG
from Solver import Solver

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


#generators = [LEHMER_0, LEHMER_1, LEHMER_2, LEHMER_3, LEHMER_4, LEHMER_5, MRG, ICG, MIXED_CG, EICG]
generators = [LEHMER_1]

def main():
    vrp_set_path = './Vrp-Set-E'
    problem_path = os.path.join(vrp_set_path, 'E-n51-k5.vrp')
    rnd=None
    for generator in generators:
        solver = Solver(problem_path, generator)
        print(solver.S)
        prob_list=[]
        for probability in range(80,100):
            prob_sublist=[]
            for _ in range(10):
                rnd = next(prng(1, solver.prng_type, rnd))
                prob_sublist+=[solver.binary_cws(rnd=rnd, probability=probability)[0]]
            prob_list+=[prob_sublist]
            print(probability,min(prob_sublist))
        #print(prob_list)
        #cost, route_list = solver.binary_cws_mcs(n=100)
        print(generator)
        print("cost: ",cost)
        print("route: ", route_list)
        print("------------------------------")


main()
