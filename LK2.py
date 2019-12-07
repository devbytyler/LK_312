import numpy as np

class LK2:
    def __init__(self, cities, bssf):
        self.cities = cities
        self.distance_matrix = [[y.costTo(x) for x in self.cities] for y in self.cities]
        self.bssf = bssf

    def get_tour_edge(self, i):
        t1 = self.bssf.route[i]
        t2 = self.bssf.route[i + 1 if i + 1 < len(self.bssf.route) else 0]
        distance = t1.costTo(t2)
        return distance, t1, t2

    def get_route_city(self, i):
        return self.bssf.route[i % len(self.bssf.route)]

    def get_tour_edges(self):
        tour_edges = []
        for i, city in enumerate(self.bssf.route):
            tour_edges.append(self.get_tour_edge(i))
        return tour_edges

    def get_non_tour_edges(self, all_edges, tour_edges):
        non_tour_edges = set(all_edges) - set(tour_edges)
        return non_tour_edges

    def get_all_edges(self):
        edges = []
        for i, row in enumerate(self.distance_matrix):
            for j, cost in enumerate(row):
                if cost < np.inf:
                    edges.append((cost, self.cities[i], self.cities[j]))
        return edges

    def solve(self):
        all_edges = self.get_all_edges()
        tour_edges = self.get_tour_edges()
        non_tour_edges = list(self.get_non_tour_edges(all_edges, tour_edges))

        for x1 in tour_edges:  # each possible x1
            best_gain = 0, None
            for y1 in [edge for edge in non_tour_edges if edge[1] == x1[2] and edge[2] != x1[1]]:  # loop over all possible y1's
                for x2 in [edge for edge in tour_edges if edge[1] == y1[2] and edge[2] != x1[1]]:  # loop over all possible x2's
                    for y2 in [edge for edge in non_tour_edges if edge[1] == x2[2] and edge[2] == x1[1]]: # only check y2's that connect back to t1
                        this_gain = y1[0] - x1[0]
                        if this_gain > best_gain[0]:
                            best_gain = this_gain, y1
            if best_gain[1]:
                pass

        '''
            loop every city
            find an x1
            from the neighbor find gain of each edge not in bssf, follow the best one (find y1)
            choose a t4 by looping over the edges not in bssf and finding one that connect to t1
            
            
            choose t1
            choose t2 (choosing x1)
            choose t3 (choosing y1) : 
                y1 is not in the tour,
                loop through all possible t4's that connect to t1
                choose the best y1 of those which gives new t4 and x2
            
        '''