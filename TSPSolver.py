#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
from types import SimpleNamespace
from LK import LinKernighan
from LK2 import LK2
import itertools


'''
Class Node represents a state in the branch and bound tree.
'''
class Node:
	FAVOR_FACTOR = 0.9  # increases priority by reducing bound

	# Creates a new node and calculates the lower bound.
	# Runs in O(n^4) time.
	# Space complexity of a single node is O(n) + O(n^2) = O(n^2).
	def __init__(self, parent, child, is_root=False):
		self.path = list(parent.path)  # space of O(n)
		self.rcm = np.array(parent.rcm)  # space of O(n^2)
		self.bound = parent.bound
		self.path.append(child)
		if not is_root:
			self.compute_enter_node()  # Runs in O(n) time.
			if self.bound == np.inf:
				return
		self.reduce_matrix()  # Runs in O(n^4) time.

	# Computes the cost of entering a node.
	# Sets corresponding row and column to infinity.
	# Runs in O(2n), which is the length of a column * the length of a row.
	def compute_enter_node(self):
		row = self.path[-2]._index
		column = self.path[-1]._index
		self.bound += self.rcm[row][column]
		self.rcm[row, :] = np.inf
		self.rcm[:, column] = np.inf
		self.rcm[column][row] = np.inf

	# Reduces the matrix by ensuring the min cost of entering and exiting every
	# node is accounted for and added to the bound.
	# Total time in O(2n^4) as explained below.
	def reduce_matrix(self):
		# gets the min element index in each row. O(n^2)
		# updates the bound and set the row to infinity.
		# Total time is O(n^4)
		row_mins = np.argmin(self.rcm, axis=1)
		for row, col in np.ndenumerate(row_mins):  # Loops over each minimum O(n). Total O(n^2)
			min_val = self.rcm[row][col]
			if min_val != 0 and min_val != np.inf:
				self.bound += min_val  # Increases lower bound
				self.rcm[row, :] -= min_val  # Sets the row to infinity O(n).

		# gets the min element index in each column.
		# updates the bound and set the column to infinity.
		# Total time is O(n^4)
		col_mins = np.argmin(self.rcm, axis=0)
		for col, row in np.ndenumerate(col_mins):  # Loops over each minimum O(n)
			min_val = self.rcm[row][col]
			if min_val != 0 and min_val != np.inf:
				self.bound += min_val  # Increases lower bound
				self.rcm[:, col] -= min_val  # O(n)

	# Returns the priority for the Node. Constant O(1)
	def get_priority(self):
		priority = self.bound * pow(self.FAVOR_FACTOR, len(self.path))
		return priority

	# Expands a node by returning an array of nodes for every city not already in the path.
	# Creating a node costs O(n^4) and a child can expand into n children, so total
	# cost is O(n^5).
	def expand_children(self, cities):
		other_cites = set(cities) - set(self.path)
		return [Node(self, city) for city in other_cites]


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	'''
	Loops though the remaining cities not already in the path and finds the closest.
	Returns the index of the closest city.
	Visits at most every city once so runs in O(n) time.
	'''
	def choose_best_neighbor(self, city, cities_left):
		best_neighbor = {
			'cost': np.inf,
			'idx': None
		}
		for i, neighbor in enumerate(cities_left):
			cost = city.costTo(neighbor)
			if cost < best_neighbor['cost']:
				best_neighbor['cost'] = cost
				best_neighbor['idx'] = i
		return best_neighbor['idx']

	'''
	Finds a greedy cycle by repeatedly finding the closest neighbor to the last city visited.
	Goes until all cities have been visited and added to the path.
	Returns a TSPSolution or None if path doesn't reconnect.
	
	Space complexity is O(n), stores a path as long as n.
	Time complexity is O(n^2). Finds the closest city (O(n)) n times.
	'''
	def findGreedyCycle(self, start_idx, cities):
		cities_left = list(cities)
		route = [cities_left.pop(start_idx)]
		while cities_left:  # Runs n times
			next_idx = self.choose_best_neighbor(route[-1], cities_left)  # Runs in O(n)
			if next_idx is None:  # city has no unvisited neighbors
				return None
			route.append(cities_left.pop(next_idx))
		if route[-1].costTo(route[0]) < np.inf:  # check end to start
			return TSPSolution(route)
		else:
			return None

	'''
	Finds a greedy cycle starting at every city in the graph. Chooses the best.
	Time complexity is O(n). Stores a bssf with n cities.
	Time complexity in O(n^3). Finds a greedy cycle (O(n^2)) n times.
	'''
	def greedy( self, time_allowance=30.0 ):
		results = {}
		cities = self._scenario.getCities()
		count = 0
		bssf = None

		start_time = time.time()
		for i, city in enumerate(cities):
			if time.time() - start_time > time_allowance:
				break
			soln = self.findGreedyCycle(i, cities)  # Runs in O(n^2)
			if soln:
				count += 1
				if not bssf or soln.cost < bssf.cost:
					bssf = soln
		end_time = time.time()

		results['cost'] = bssf.cost if count else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	'''
	Uses branch and bound to find an optimum cycle.
	Total time cost breakdown:
	+ O(n^2) for distance matrix init
	+ O(n^3) for initial greedy BSSF
	+ O(n^4) for root node creation
	+ O(log k) for root node heap insertion
	+ O(k) * O(n^5 + n log k) for adding and removing to the heap until empty.
	= I don't know what that works out to be, but it's exponential in the worst case. 
	It's something like O(n^5 b^n nlog k) where b is the branching factor.
	'''
	def branchAndBound(self, time_allowance=60.0):
		results = {}
		heap = []
		count = 0
		cities = self._scenario.getCities()
		# Calculates the initial distance matrix. Assuming costTo() is constant, building the
		# distance matrix is O(n^2) time and space.
		distance_matrix = np.array([np.array([y.costTo(x) for x in cities]) for y in cities])
		start_time = time.time()
		bssf = self.greedy(time_allowance)['soln']  # Uses greedy() to find initial bssf. Costs O(n^3)
		pruned = 0
		total = 0
		max_queue = 0

		root_data = {
			'path': [],
			'rcm': distance_matrix,
			'bound': 0,
		}

		# Creates the first state in the tree. Costs O(n^4) time.
		root_node = Node(SimpleNamespace(**root_data), cities[0], is_root=True)
		# Pushes the root node onto the heap. Heap insertions cost O(log k), k = size of heap.
		heapq.heappush(heap, (root_node.get_priority(), id(root_node), root_node))
		while len(heap) and time.time()-start_time < time_allowance:
			# Pops the first node from the heap. Heap removal costs O(log k), k = size of heap.
			node = heapq.heappop(heap)[2]
			if node.bound >= bssf.cost:  # prune node enqueued node if bssf has changed
				pruned += 1
			else:
				children = node.expand_children(cities)  # Costs O(n^5) time, and O(n^3) space.
				total += len(children)
				if len(children) == 0 and node.bound < bssf.cost:  # The current node is a leaf node.
					count += 1
					bssf = TSPSolution(node.path)
				else:
					for child in children:  # Costs at most O(n log k), a heap insertion for each city.
						if child.bound < bssf.cost:
							# Pushes the child node onto the heap. Heap insertions cost O(log k), k = size of heap.
							heapq.heappush(heap, (child.get_priority(), id(child), child))
							if len(heap) > max_queue:
								max_queue = len(heap)
						else:
							pruned += 1
		end_time = time.time()
		pruned += len(heap)  # Include nodes never removed from the queue as per spec

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = max_queue
		results['total'] = total
		results['pruned'] = pruned
		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	def fancy(self, time_allowance=60.0):
		bssf = self.greedy(time_allowance)['soln']
		lk = LK2(self._scenario.getCities(), bssf)
		result = lk.solve()
		return result





