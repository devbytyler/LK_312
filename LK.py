
import numpy as np
from TSPClasses import *

class LinKernighan:

    def __init__(self, cities):
        self.cities = cities

    def get_diffs(self, city, set_a, set_b, distance_matrix, cities):
        external = 0
        internal = 0
        if city in set_a:
            for i, neighbor_cost in enumerate(distance_matrix[city._index]):
                if neighbor_cost != np.inf:
                    if cities[i] in set_a:
                        internal += neighbor_cost
                    else:
                        external += neighbor_cost
        else:
            for i, neighbor_cost in enumerate(distance_matrix[city._index]):
                if neighbor_cost != np.inf:
                    if cities[i] in set_b:
                        internal += neighbor_cost
                    else:
                        external += neighbor_cost
        return external - internal

    def split_graph(self, cities):
        middle = len(cities) // 2
        return cities[:middle], cities[middle:]

    def get_max_gain(self, diffs):
        return None, None, None

    def solve(self):
        distance_matrix = np.array([np.array([y.costTo(x) for x in self.cities]) for y in self.cities])
        set_a, set_b = self.split_graph(self.cities)

        while True:
            diffs = [self.get_diffs(city, set_a, set_b, distance_matrix, self.cities) for city in self.cities]
            av, bv, gv = [], [], []
            univisited = list(self.cities)
            for i in range():
                a, b, g = self.get_max_gain(diffs)
                # remove a and b from unvisted
                av.append(a)
                bv.append(b)
                gv.append(g)
                # update diffs

