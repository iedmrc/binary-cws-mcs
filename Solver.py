import os
import itertools
from threading import Thread
import concurrent
import concurrent.futures
import copy

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
        S = self.get_savings_list(self.M)

        # Sorted saving list
        self.S = S[(-S[:, 2]).argsort()]

        # Random number generator type
        self.prng_type = prng_type

    def get_savings_list(self, M):
        S = []
        n = len(M)

        for i in range(1, n):
            for j in range(1, i):
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
        if not len(set(s[0:2]) & set(itertools.chain.from_iterable(route_list))):
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

    def spread_missing_nodes(self, route_list):
        all_nodes = set(self.problem.get_nodes())
        current_nodes = set(itertools.chain.from_iterable(route_list))

        missing_nodes = all_nodes - current_nodes
        missing_nodes.remove(0)

        # If there are missing nodes
        if len(missing_nodes):
            # then add them without checking capacity
            route_list.append(list(missing_nodes))

    def cost(self, route_list):
        cost = 0
        for route in route_list:
            cost += self.M[0][route[0]] + self.M[route[-1]][0]
            for i in range(len(route)-1):
                cost += self.M[route[i]][route[i+1]]
        return cost

    def construct_pivot_list(self, savings):
        return list(range(len(savings)))

    def binary_cws(self, route_list=[], savings=np.array([]), rnd=None):
        savings = savings if len(savings) else self.S
        pivot_list = self.construct_pivot_list(savings)
        route_list = copy.deepcopy(route_list)

        rnd = next(prng(1, self.prng_type, rnd))[0]
        probability = rnd % 40

        while len(pivot_list) > 0:
            pivot_list_helper = pivot_list.copy()
            for i in pivot_list_helper:
                rnd = next(prng(1, self.prng_type, rnd))[0]
                if rnd % 100 < probability:
                    print(i)
                    self.process(savings[i], route_list)
                    pivot_list.remove(i)

        self.spread_missing_nodes(route_list)

        return self.cost(route_list), route_list

    def binary_cws_mcs(self, n=100):
        route_list = []
        savings = self.S.copy()

        pivot_list = self.construct_pivot_list(savings)

        while len(pivot_list) > 0:
            pivot_list_helper = pivot_list.copy()
            for i in pivot_list_helper:
                print(i)
                t1, t2 = [], []
                for seeder in prng(n):
                    current_step = self.binary_cws(
                        route_list=route_list, savings=savings[i:], rnd=seeder[0])
                    t1.append(current_step[0])

                    next_step = self.binary_cws(
                        route_list=route_list, savings=savings[i+1:], rnd=seeder[0])
                    t2.append(next_step[0])
                print(sum(t2)/n ,sum(t1)/n)
                if sum(t2)/n >= sum(t1)/n:
                    self.process(savings[i], route_list)
                    pivot_list.remove(i)

        self.spread_missing_nodes(route_list)

        return self.cost(route_list), route_list
