"""
Container package for Solver Class
"""
import os
import itertools
import copy
import time

import tsplib95
import networkx as nx
import numpy as np
import ray
from tqdm import tqdm

from prng import prng, LEHMER_0


class Solver:
    """
    A wrapper class for Binary-CWS and Binary-CWS-MCS vehicle routing solver algorithms.
    """

    def __init__(self, problem_path, prng_type=LEHMER_0):
        # Problem instance
        self.problem = tsplib95.load_problem(problem_path)

        # Graph of nodes
        G = self.problem.get_graph()

        # Cost (distance/duration) matrix
        self.M = np.array(nx.to_numpy_matrix(G))

        # Depot index
        if len(self.problem.depots) > 1:
            raise Exception(
                "This solver can only work with one depot problems. Please make sure that the problem only has one depot.")

        self.DI = self.problem.depots[0] - 1

        # Capacity constraint of vehicles
        self.CAPACITY = self.problem.capacity

        # Demands of jobs
        self.DEMANDS = [v for k, v in self.problem.demands.items()]

        # The savings list
        S = self._construct_savings_list(self.M)

        # Sorted saving list
        self.S = S[(-S[:, 2]).argsort()]

        # Random number generator type
        self.prng_type = prng_type

    @staticmethod
    def _construct_savings_list(M):
        """
        Builds savings list in order to be processed in Clarke & Wright’s Savings Algorithm.

        Takes, Frank W., and Walter A, Kosters. "Applying Monte Carlo Techniques 
        to the Capacitated Vehicle Routing Problem." 22th Benelux
        Confenrece On Artificial Intelligence (BNAIC). 2010.

        Parameters
        ----------
        M : Distance (Cost) matrix. A list of lists such that M[i][j]=distance(i,j) 
        for every i,j = 1,...,n and n is the number of nodes (jobs).


        Returns
        -------
        S : Two dimensional numpy array of savings matrix

        """
        S = []
        n = len(M)

        for i in range(1, n):
            for j in range(1, i):
                if i != j:
                    S_ij = M[0][i] + M[0][j] - M[i][j]
                    S += [[i, j, S_ij]]

        return np.array(S, dtype='int')

    def _check_capacity(self, route, new_jobs):
        """
        Checks whether the capacity of route is welcome for the new nodes to be added, or not.

        Parameters
        ----------
        route : List of current route
            
        new_jobs : List of new nodes to be added, if they pass capacity check
            

        Returns
        -------
        Boolean : True if capacity is not full else False

        """
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

    def _process(self, s, route_list):
        """
        Processes the node in s, according to Clarke & Wright’s Savings Algorithm.

        Takes, Frank W., and Walter A, Kosters. "Applying Monte Carlo Techniques 
        to the Capacitated Vehicle Routing Problem." 22th Benelux
        Confenrece On Artificial Intelligence (BNAIC). 2010.

        Parameters
        ----------
        s : A savings element which is type of (i,j,s) tuple such that i,j = 1,...,n
        and n is the number of nodes (jobs).
            
        route_list : List of routes
            

        Returns
        -------
        No value. Mutates the route_list accordingly.

        """
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
            if self._check_capacity([], s[0:2]):
                route_list.append(s[0:2].tolist())
        # If both i and j exist at the margin of two routes
        elif len(location_i) and len(location_j):
            # and if these routes are distinct
            if location_i[0] != location_j[0]:
                # then merge these routes
                route1 = route_list[location_i[0]]
                route2 = route_list[location_j[0]]
                merged_route = []

                if not self._check_capacity(route1, route2):
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
            if not self._check_capacity(route_list[location_i[0]], s[1]):
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
            if not self._check_capacity(route_list[location_j[0]], s[0]):
                return

            # If j exists at the beginning of a route
            if location_j[1] == 0:
                # then prepend i to that route
                route_list[location_j[0]].insert(0, s[0].tolist())
            # If j exists at the end of a route
            elif location_j[1] == -1:
                # then append i to that route
                route_list[location_j[0]].append(s[0].tolist())

    def _spread_missing_nodes(self, route_list):
        """
        Fills route_list with the nodes missing in it, just by assigning them as a new route.

        Parameters
        ----------
        route_list : List of routes
            

        Returns
        -------
        No value. Mutates the route_list accordingly.

        """
        all_nodes = set(range(len(self.M)))
        current_nodes = set(itertools.chain.from_iterable(route_list))
        depot_node = set([self.DI])
        # print(all_nodes, current_nodes, depot_node)
        missing_nodes = all_nodes - current_nodes - depot_node

        # If there are missing nodes
        if len(missing_nodes):
            # then add them without checking capacity
            route_list.append(list(missing_nodes))

    def _cost(self, route_list):
        """
        Calculates the cost of the route_list.

        Parameters
        ----------
        route_list : List of routes
            

        Returns
        -------
        cost : Integer value of cost

        """
        cost = 0
        for route in route_list:
            # Calculate travelling from/to depot cost
            cost += self.M[self.DI][route[0]] + self.M[route[-1]][self.DI]
            # Calculate each vehicle's travel cost
            for i in range(len(route)-1):
                cost += self.M[route[i]][route[i+1]]
        return cost

    @staticmethod
    def _construct_pivot_list(savings):
        """
        Returns a list with the same length of savings.

        Parameters
        ----------
        savings : Savings list
            

        Returns
        -------
        List

        """
        return list(range(len(savings)))

    @staticmethod
    def _fork(data):
        """
        Returns a deepcopy of the data.

        Parameters
        ----------
        data : Any list
            

        Returns
        -------
        List

        """
        return copy.deepcopy(data)

    def binary_cws(self, **kwargs):
        """
        Implementation of the Binary CWS vehicle routing solver algorithm.
        Returns the cost and the routes computed.

        Takes, Frank W., and Walter A, Kosters. "Applying Monte Carlo Techniques 
        to the Capacitated Vehicle Routing Problem." 22th Benelux
        Confenrece On Artificial Intelligence (BNAIC). 2010.

        Parameters
        ----------
        **kwargs:

        seeder : Seeder of the pseudo-random number generator.
             (Default value = None)

        probability: Probability of processing the current saving.
             (Default value = 40)

        savings: Savings list
             (Default value = self.S)

        route_list: List of routes
             (Default value = [])


        Returns
        -------
        cost, route_list : The cost and the routes.

        """
        r = kwargs.get('seeder', None)
        p = kwargs.get('probability', 40)
        savings = kwargs.get('savings', self.S)

        if 'route_list' in kwargs:
            route_list = self._fork(kwargs['route_list'])
        else:
            route_list = []

        pivot_list = self._construct_pivot_list(savings)

        while len(pivot_list) > 0:
            for i in self._fork(pivot_list):
                r = next(prng(self.prng_type, z=r))
                if r % 100 >= p:
                    self._process(savings[i], route_list)
                    pivot_list.remove(i)

        self._spread_missing_nodes(route_list)
        cost = self._cost(route_list)

        return cost, route_list

    def _bulk_binary_cws(self, random_numbers, route_list, savings_current, savings_next):
        """
        Runs binary_cws method len(random_numbers) times with the current and next savings.
        Returns a list of costs as t1, t2 with respect to savings_current and savings_next.

        Parameters
        ----------
        random_numbers : Random numbers will be passed in turn, to binary_cws as seeder.
            
        route_list : List of routes
            
        savings_current : Savings list of current step.
            
        savings_next : Savings list of next (i+1) step.
            

        Returns
        -------
        t1_jobs, t2_jobs : A tuple of a list of costs as t1, t2 with respect to 
        savings_current and savings_next.

        """
        t1_jobs, t2_jobs = [], []

        for random_number in random_numbers:
            probability = (random_number % 21) + 5

            t1_job = self.binary_cws(
                route_list=route_list, savings=savings_current, seeder=random_number, probability=probability)
            t1_jobs.append(t1_job[0])

            t2_job = self.binary_cws(
                route_list=route_list, savings=savings_next, seeder=random_number, probability=probability)
            t2_jobs.append(t2_job[0])

        return t1_jobs, t2_jobs

    @ray.remote
    def _chunked_binary_cws(self, random_numbers, route_list, savings_current, savings_next):
        """
        A ray.remote decorated method that distributes chunks of binary_cws to available processors.
        With the help of this method, we can employ multiple processes in order to
        run binary_cws in parallel.

        Parameters
        ----------
        random_numbers : Random numbers will be passed in turn, to binary_cws as seeder.
            
        route_list : List of routes
            
        savings_current : Savings list of current step.
            
        savings_next : Savings list of next (i+1) step.
            

        Returns
        -------
        _bulk_binary_cws

        """
        return self._bulk_binary_cws(random_numbers, route_list, savings_current, savings_next)

    @staticmethod
    def _create_chunks(list_name, n):
        """
        Create chunks of the given list

        Parameters
        ----------
        list_name : A list to be chunked
            
        n : Chunk size
            

        Returns
        -------
        Generator: Chunked lists

        """
        for i in range(0, len(list_name), n):
            yield list_name[i:i + n]

    def _monte_carlo(self, route_list, savings_current, savings_next, **kwargs):
        """
        Runs monte carlo simulation of binary_cws as n times.

        Parameters
        ----------
        route_list : List of routes
            
        savings_current : Savings list of current step.
            
        savings_next : Savings list of next (i+1) step.
            
        **kwargs :

        n : Monte Carlo simulation number
             (Default value = 100)

        seeder : Seeder of the pseudo-random number generator.
             (Default value = None)

        distributed: Boolean value of whether simulation will be distributed to processors or not.
             (Default value = False)
            

        Returns
        -------
        t1, t2 : A tuple of a mean of costs as t1, t2 with respect to 
        savings_current and savings_next.

        """
        n = kwargs.get('n', 100)
        seeder = kwargs.get('seeder', None)
        distributed = kwargs.get('distributed', False)

        if distributed:
            cores = ray.resource_spec.multiprocessing.cpu_count()
            random_number_chunks = self._create_chunks(
                list(prng(self.prng_type, n, seeder)), n//cores)
            jobs = []

            for random_numbers in random_number_chunks:
                jobs.append(self._chunked_binary_cws.remote(
                    self, random_numbers, route_list, savings_current, savings_next))

            t1, t2 = 0, 0

            while len(jobs):
                done, jobs = ray.wait(jobs)
                done = ray.get(done)[0]

                t1 += sum(done[0])
                t2 += sum(done[1])

            return t1/n, t2/n
        else:
            t1, t2 = self._bulk_binary_cws(
                prng(self.prng_type, n, seeder), route_list, savings_current, savings_next)
            return sum(t1)/n, sum(t2)/n

    def binary_cws_mcs(self, n=100, distributed=False):
        """
        Implementation of the Binary CWS-MCS vehicle routing solver algorithm.
        Returns the cost and the routes computed.

        Takes, Frank W., and Walter A, Kosters. "Applying Monte Carlo Techniques 
        to the Capacitated Vehicle Routing Problem." 22th Benelux
        Confenrece On Artificial Intelligence (BNAIC). 2010.

        Parameters
        ----------
        n : Monte Carlo simulation number
             (Default value = 100)
        distributed : Boolean value of whether simulation will be distributed to processors or not.
             (Default value = False)


        Returns
        -------
        cost, route_list : The cost and the routes.

        """
        if distributed:
            ray.init()
            while not ray.is_initialized:
                time.sleep(0.01)

        route_list = []
        savings = self.S
        pivot_list = self._construct_pivot_list(savings)
        pbar = tqdm(total=len(pivot_list), desc="Binary CWS-MCS")

        while len(pivot_list) > 0:
            for i in self._fork(pivot_list):
                savings_current = savings[i:]
                savings_next = savings[i+1:]
                t1, t2 = self._monte_carlo(
                    route_list, savings_current, savings_next, n=n, distributed=distributed)

                if t2 >= t1:
                    self._process(savings[i], route_list)
                    pivot_list.remove(i)
                    pbar.update(1)

        self._spread_missing_nodes(route_list)
        cost = self._cost(route_list)
        pbar.close()

        return cost, route_list
