import os

from prng import prng, LEHMER_0, LEHMER_1, LEHMER_2, LEHMER_3, LEHMER_4, LEHMER_5, MRG, ICG, MIXED_CG, EICG
from Solver import Solver

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


#generators = [LEHMER_0, LEHMER_1, LEHMER_2, LEHMER_3, LEHMER_4, LEHMER_5, MRG, ICG, MIXED_CG, EICG]
generators = [MRG]


def main():
    vrp_set_path = './Vrp-Set-E'
    
    rnd = None
    for vrp in ['E-n13-k4.vrp']:#,'E-n22-k4.vrp','E-n23-k3.vrp','E-n30-k3.vrp','E-n31-k7.vrp','E-n33-k4.vrp','E-n51-k5.vrp','E-n76-k7.vrp','E-n76-k8.vrp','E-n76-k10.vrp','E-n76-k14.vrp','E-n101-k8.vrp','E-n101-k14.vrp']:
        problem_path = os.path.join(vrp_set_path, vrp)
        print("problem_path:",problem_path)
        for generator in generators:
            print("generator:",generator)
            solver = Solver(problem_path, generator)
            
            #prob_list = []
            #for probability in range(1,100):
                #prob_sublist=[]
                #for _ in range(100):
                    #rnd = next(prng(1, solver.prng_type, rnd))
                    #prob_sublist+=[solver.binary_cws(rnd=rnd, probability=probability)[0]]
                #prob_list+=[prob_sublist]
                #print(probability,min(prob_sublist),sum(prob_sublist)/10**2)
            #print(prob_list)

            cost, route_list = solver.binary_cws_mcs(n=10)
            print("cost: ", cost)
            #print("route: ", route_list)
        # route_list = [[17, 20, 18, 15, 12],[16, 19, 21, 14] ,[13, 11, 4, 3, 8, 10] ,[9, 7, 5, 2, 1, 6]]


main()
