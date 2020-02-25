import os
import itertools

import tsplib95
import networkx as nx
import numpy as np

from prng import prng, LEHMER_0

class Solver:
    def __init__(self, problem_path, prng_type=LEHMER_0):
        # Problem instance
        self.problem = tsplib95.load_problem(problem_path)

        # Graph of nodes
        self.G = self.problem.get_graph()

        # Cost (distance/duration) matrix
        self.M = np.array(nx.to_numpy_matrix(self.G))

        # Depot index
        self.DI = self.problem.depots[0] - 1

        # Capacity constraint of vehicles
        self.CAPACITY = self.problem.capacity

        # Demands of jobs
        self.DEMANDS = [v for k, v in self.problem.demands.items()]

        # The saving list
        S = self.get_saving_list(self.M)

        # Sorted saving list
        self.S = S[(-S[:, 2]).argsort()]

        # Random number generator type
        self.prng_type = prng_type

    def get_saving_list(self, M):
        S = []
        n = len(M)

        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    S_ij = M[0][i] + M[0][j] - M[i][j]
                    S += [[i, j, S_ij]]

        return np.array(S, dtype='int')


    def check_capacity(self, route, new_jobs):
        capacity = 0
        if type(new_jobs).__module__ == np.__name__:
            if new_jobs.size == 1:
                new_route = route + [int(new_jobs)]
            elif new_jobs.size > 1:
                new_route = route + new_jobs.tolist()
        else:
            new_route = route + new_jobs

        for job in new_route:
            capacity += self.DEMANDS[job]
            if capacity > self.CAPACITY:
                return False
        
        return True


    def process(self, s, route_list):
        location_i, location_j = [], []
        # Spot the location of i and j in route_list
        for idx, route in enumerate(route_list):
            if s[0] == route[0]:
                location_i = [idx, 0]
            elif s[0] == route[-1]:
                location_i = [idx, -1]

            if s[1] == route[0]:
                location_j = [idx, 0]
            elif s[1] == route[-1]:
                location_j = [idx, -1]

        # If neither i nor j assigned to a route in route_list
        if not len(set(s[0:2]) & set(list(itertools.chain.from_iterable(route_list)))):
            # then initiate a new route with i,j
            if self.check_capacity([], s[0:2]):
                route_list.append(s[0:2].tolist())
        # If both i and j exist at the margin of two routes
        elif len(location_i) and len(location_j):
            # and if these routes are distinct
            if location_i[0] != location_j[0]:
                # then merge these routes
                route1 = route_list[location_i[0]]
                route2 = route_list[location_j[0]]
                merged_route = []

                if not self.check_capacity(route1, route2):
                    return

                if location_i[1] == location_j[1]:
                    if location_i[1] == 0:
                        merged_route = route1[::-1] + route2
                    elif location_i[1] == -1:
                        merged_route = route1 + route2[::-1]
                elif location_i[1] == 0 and location_j[1] == -1:
                    merged_route = route2 + route1
                elif location_i[1] == -1 and location_j[1] == 0:
                    merged_route = route1 + route2

                indices = [location_i[0], location_j[0]]
                for i in sorted(indices, reverse=True):
                    del route_list[i]
                route_list.append(merged_route)
            # If these routes are the same
            else:
                # then do nothing
                return
        # If i exists at the margin of a route
        elif len(location_i) != 0 and s[1] not in list(itertools.chain.from_iterable(route_list)):
            if not self.check_capacity(route_list[location_i[0]], s[1]):
                return
            
            # If i exists at the beginning of a route
            if location_i[1] == 0:
                # then prepend j to that route
                route_list[location_i[0]].insert(0, s[1].tolist())
            # If i exists at the end of a route
            elif location_i[1] == -1:
                # then append j to that route
                route_list[location_i[0]].append(s[1].tolist())
        # If j exists at the margin of a route
        elif len(location_j) != 0 and s[0] not in list(itertools.chain.from_iterable(route_list)):
            if not self.check_capacity(route_list[location_j[0]], s[0]):
                return
            
            # If j exists at the beginning of a route
            if location_j[1] == 0:
                # then prepend i to that route
                route_list[location_j[0]].insert(0, s[0].tolist())
            # If j exists at the end of a route
            elif location_j[1] == -1:
                # then append i to that route
                route_list[location_j[0]].append(s[0].tolist())


    def cost(self, route_list):
        cost = 0
        for route in route_list:
            cost += self.M[0][route[0]] + self.M[route[-1]][0]
            for i in range(len(route)-1):
                cost += self.M[route[i]][route[i+1]]
        return cost


    def binary_cws(self, k=0):
        route_list = []
        pivot_list = list(range(k, len(self.S)))
        rnd = None

        while len(pivot_list) > 0:
            for i in pivot_list:
                rnd = next(prng(1, self.prng_type, rnd))[0]
                if 5 < rnd % 100 < 40:
                    self.process(self.S[i], route_list)
                    pivot_list.remove(i)
        
        return self.cost(route_list), route_list