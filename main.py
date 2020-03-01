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
    
    rnd = None
    for generator in generators:
        solver = Solver(problem_path, generator)
        prob_list = []
        for probability in range(1,100):
            prob_sublist=[]
            for _ in range(10**2):
                rnd = next(prng(1, solver.prng_type, rnd))
                prob_sublist+=[solver.binary_cws(rnd=rnd, probability=probability)[0]]
            prob_list+=[prob_sublist]
            print(probability,min(prob_sublist),sum(prob_sublist)/10**2)
        #print(prob_list)
    # print("cost: ", cost)
    # print("route: ", route_list)
    # route_list = [[17, 20, 18, 15, 12],[16, 19, 21, 14] ,[13, 11, 4, 3, 8, 10] ,[9, 7, 5, 2, 1, 6]]


main()
