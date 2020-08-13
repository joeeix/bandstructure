import numpy as np
from .lattice import Lattice


class TriangularLattice(Lattice):
	def initialize(self):
		l1 = np.array([1, 0])
		l2 = np.array([1,np.sqrt(3)])/2
		
		self.addLatticevector(l1)
		self.addLatticevector(l2)
		self.addBasisvector([0, 0])
