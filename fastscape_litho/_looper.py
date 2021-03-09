"""
TAKEN FROM lsdnumbatools, temporary measure before the latter is available
this provides a numba jitclass to loop through node effiviently

B.G.
"""
import numpy as np
import numba as nb
import math as m
from numba.experimental import jitclass

@nb.njit(inline='always')
def _get_neighbouring_nodes_with_checks(output,adder,node):
	i = 0
	for add in adder:
		if(add != -2147483648):
			output[i] = node + add
		else:
			output[i] = -2147483648
		i += 1


# specs for the jitclass which needs to be statically typed
spec = [
		('node_type', nb.int16[:]),
		('nx', nb.int32),
		('ny', nb.int32),
		('nxy', nb.int32),
		('dx', nb.float64),
		('dy', nb.float64),
		('dxy', nb.float64),
		('nodata', nb.int32),
		('neighbourer', nb.int32[:,:]),
		('lengthener', nb.float64[:]),
	]
	#min, max int32 = -2,147,483,648 2,147,483,647

@jitclass(spec)
class iterasator(object):
	"""
		The iterasator class is made to return neighboring informations from a node.
		It takes care of boundary conditions for example, where the geometric neighbors are not the right ones.
		Date: November 2020
		Authors: B.G.
	"""
	def __init__(self,node_type,nx,ny, dx, dy):
		"""
		Initialises the iterasator. 
		node_type (1D numpy array of int): vectorised array of node type
		nx (int): number of cols
		ny (int): number of rows
		Date: November 2020
		Authors: B.G.
		"""
		# The nodes type are documented in the file node_graph.py
		self.node_type = node_type
		# Number of cols
		self.nx = nx
		# Number of rows
		self.ny = ny

		#number of elements
		self.nxy = nx*ny

		# No data value for the neighbourer
		self.nodata = -2147483648

		self.dx = dx
		self.dy = dy
		self.dxy = m.sqrt(dx**2 + dy**2)
		# The neighbourer is the most important (and irksome to code) structure of this class:
		# It hardcodes the window looping through vectorised neighbours for each cases
		# I adopted this approach rather than padding no data to be as flexible as possible
		self.neighbourer = np.array([
[ - nx - 1, - nx, - nx + 1, -1, 1, nx - 1, nx, nx + 1], # neighbourer 0: classic internal D8 node and all the neighbors are returned
[ - 1, - nx, - nx + 1, nx - 1, 1, 2 * nx - 1, nx, nx + 1], # neighbourer 1: periodic D8 node at the Western Boundary
[ - 1 + ny * nx,  ny * nx, ny * nx + 1, -1, 1, nx - 1, nx, nx + 1], # neighbourer 2: periodic D8 node at the Northern Boundary
[ - nx - 1, - nx, - 2 * nx + 1, -1, - nx + 1, nx - 1, nx, 1], # neighbourer 3: periodic D8 node at the Eastern Boundary
[ - nx - 1, - nx, - nx + 1, -1, 1, - ny * nx - 1, - ny * nx, - ny * nx + 1], # neighbourer 4: periodic D8 node at the Southern Boundary
[ self.nodata, self.nodata, self.nodata, -1, 1, nx - 1, nx, nx + 1], # neighbourer 5: classic D8 node Northern boundary
[ self.nodata, - nx, - nx + 1, self.nodata, 1, self.nodata, nx, nx + 1], # neighbourer 6: classic D8 node Western boundary
[ - nx - 1, - nx, - nx + 1, -1, 1, self.nodata, self.nodata, self.nodata], # neighbourer 7: classic D8 node Southern boundary
[ - nx - 1, - nx, self.nodata, -1, self.nodata, nx - 1, nx, self.nodata], # neighbourer 8: classic D8 node Eastern boundary
[ self.nodata,  self.nodata, self.nodata, self.nodata, 1, self.nodata, nx, nx + 1], # neighbourer 9: classic D8 node at the Northern-western corner Boundary
[ self.nodata, - nx, - nx + 1, self.nodata, 1, self.nodata, self.nodata, self.nodata], # neighbourer 10: classic D8 node at the Southwestern corner Boundary
[ - nx - 1, - nx, self.nodata, -1, self.nodata, self.nodata, self.nodata, self.nodata], # neighbourer 11: classic D8 node at the Southeastern corner Boundary
[ self.nodata, self.nodata, self.nodata, -1, self.nodata, nx - 1, nx, self.nodata], # neighbourer 12: classic D8 node at the Northeastern corner Boundary
# To add for the future, not urgent: 
# - Cyclic boundaries cause why not
# - specific corner boundary (e.g. NW with W periodic and NW with N periodic)
# - Internal periodic? (i.e. goes straight to the next periodic in the interested direction)
		],dtype = nb.int32)

		# The lengthener is associated with the neighbourer to return the length to the neighbour associated to each one
		self.lengthener = np.array([self.dxy,self.dy,self.dxy,self.dx,self.dx,self.dxy,self.dy,self.dxy], nb.float64)

	def node2rowcol(self,node):
		"""
		Converts vectorised node ID to row index and col index
		"""
		r = node // self.nx
		c = node % self.nx
		return r,c




	def get_neighbouring_nodes(self,node):
		"""
		return all the neighboring nodes of a diven node index
		"""
		output = np.full(8,self.nodata,dtype = nb.int32)
		# case 1: active, classic internal node, keeping it test-free to optimise code as this will be the overwhelming majority of cases
		# THE ONLY REQUIREMENT IS THAT NO BOUNDARY NODES ARE LABELLED WITH 1
		if(self.node_type[node] == 1):
			i = 0
			for add in self.neighbourer[0]:
				output[i] = node + add
				i += 1
		# North row (in array point of view)
		elif(node < self.nx):
			# classic border bourndary
			if(self.node_type[node] == 0):
				# N case
				if(node > 0 and node < self.nx - 1):
					_get_neighbouring_nodes_with_checks(output,self.neighbourer[5],node)
				# NW case
				elif (node == 0):
					_get_neighbouring_nodes_with_checks(output,self.neighbourer[9],node)
				# NE case
				elif (node == self.nx - 1):
					_get_neighbouring_nodes_with_checks(output,self.neighbourer[12],node)
			# periodic boundaries
			if(self.node_type[node] == 2):
				_get_neighbouring_nodes_with_checks(output,self.neighbourer[2],node)
		# South row 
		elif(node >= (self.nx * self.ny) - self.nx):
			# classic border bourndary
			if(self.node_type[node] == 0):
				# Middle of the S row
				if(node > (self.nx * self.ny - self.nx) and node < (self.nx * self.ny - self.nx) - 1):
					_get_neighbouring_nodes_with_checks(output,self.neighbourer[7],node)
				# SW classic boulder
				elif (node == (self.nx * self.ny - self.nx)):
					_get_neighbouring_nodes_with_checks(output,self.neighbourer[10],node)
					# SE classic boundary
				elif (node == (self.nx * self.ny - self.nx) - 1):
					_get_neighbouring_nodes_with_checks(output,self.neighbourer[11],node)
			elif(self.node_type[node] == 2):
				_get_neighbouring_nodes_with_checks(output,self.neighbourer[4],node)
	    # first Column
		elif(node % self.nx == 0):
			if(self.node_type[node] == 0):
				_get_neighbouring_nodes_with_checks(output,self.neighbourer[6],node)
			elif(self.node_type[node] == 2):
				_get_neighbouring_nodes_with_checks(output,self.neighbourer[1],node)
	    # Last Column
		elif(node % self.nx - 1 == 0):
			if(self.node_type[node] == 0):
				_get_neighbouring_nodes_with_checks(output,self.neighbourer[8],node)
			elif(self.node_type[node] == 2):
				_get_neighbouring_nodes_with_checks(output,self.neighbourer[3],node)

		return output



























#End of fucking file (I am extremely frustrated at Sublime text which delete the last empty rows of a file automatically and wants you to then work using obly the absolute bottom of your screen. I am not alone right?)
