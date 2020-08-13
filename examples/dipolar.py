#Some simple examples and tests examining dipolar systems
import sys
import numpy as np
sys.path.append("../")
from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import SquareLattice, HoneycombLattice, TriangularLattice

#Initiate Lattice object and system object with params
lattice = HoneycombLattice()
cutoff = 10
tbar = 1.
t = 0.54*tbar
w = 1.97*tbar
params = Parameters({'lattice':lattice,'cutoff':cutoff,'tbar':tbar,'w':w,'t':t})
s = DipolarSystem(params)


#Create Special Point M (lines 105-107 of lattice.py change [1,1]->[2pi/a,2pi/a])
M = np.array([0.5,0.5]) #M is in units of Reciprocal Lattic Vectors
lattice.addSpecialPoint('M',M)


#Create Rhomboidal BZ and dispersion relation as in Fig.(2(a)) of arxiv.org/pdf/1410.5667.pdf
Zone = lattice.getKvectorsRhomboid(resolution=300)
path1 = lattice.getKvectorsPath(resolution=300,pointlabels=['M','G','M'])
path2 = lattice.getKvectorsPath(resolution=300,pointlabels=['X','G','X'])
path3 = lattice.getKvectorsPath(resolution=300,pointlabels=['Y','G','Y'])

BZ = s.solve(Zone)
Disp1 = s.solve(path1)
Disp2 = s.solve(path2)
Disp3 = s.solve(path3)


#print total BerryFlux (BF) for BZ: divide by (2*np.pi) for Chern number. It seems that .getBerryFlux() returns IM(BF) already
#for i in range(BZ.numBands()):
#    BF = BZ.getBerryFlux(i)
#    print("C for band " + str(i) + ":  ",BF/(2*np.pi))


#Plot paths
Disp1.plot("path1.pdf")
Disp2.plot("path2.pdf")
Disp3.plot("path3.pdf")

