import numpy as np
import itertools

from .displacements import Displacements
from .kvectors import Kvectors


class Lattice():
    """Class to generate the lattice."""

    __vecsLattice = np.array([], dtype=np.float)
    __vecsBasis = np.array([], dtype=np.float)
    __idxBasis = np.array([])
    __idxSub = np.array([])
    __vecsReciprocal = np.array([], dtype=np.float)
    __posBrillouinZone = np.array([], dtype=np.float)
    __posBrillouinPath = np.array([], dtype=np.float)
    __specialPoints = { }

    __tol = 1e-16

    def __init__(self):
        self.initialize()

    def initialize(self):
        pass

    @property
    def vecsReciprocal(self):
        return self.__vecsReciprocal

    @property
    def vecsBasis(self):
        pass #TODO

    @vecsBasis.setter
    def vecsBasis(self, value):
        pass #TODO

    def getSpecialPoints(self, reciprocalBasis = False):
        """Return the list of userdefined and automatically generated special points that can be
        used to describe a path through the Brillouin zone ( e.g. 'G' stands for automatically
        generated gamma point)."""

        userdefinedSpecialPoints = self.__specialPoints.copy()

        automaticSpecialPoints = { }

        # === standardize the lattice vectors ===
        # --- Making certain lattice vectors and products give standard, positive values. e.g. Area  = v1xv2 ---

        if self.getDimensionality() >= 1:
            vec1 = self.__vecsReciprocal[0]
            if np.vdot(vec1,[1,0]) < 0: vec1 *= -1

        if self.getDimensionality() >= 2:
            vec2 = self.__vecsReciprocal[1]
            if np.vdot(vec1,vec2) < 0: vec2 *= -1
            if np.arctan2(vec1[1],vec1[0]) < np.arctan2(vec2[1],vec2[0]): vec1, vec2 = vec2, vec1

        # === calculate special points ===
        # --- special points for 0D and higher dimensions ---
        automaticSpecialPoints['G'] = [0, 0]

        # --- special points for 1D and higher dimensions ---
        if self.getDimensionality() >= 1:
            automaticSpecialPoints['X']      = vec1/2
            automaticSpecialPoints['-X']     = -automaticSpecialPoints['X']

        # --- special points for 2D ---
        if self.getDimensionality() >= 2:
            automaticSpecialPoints['Y']      = vec2/2
            automaticSpecialPoints['-Y']     = -automaticSpecialPoints['Y']

            automaticSpecialPoints['Z']      = (vec2-vec1)/2
            automaticSpecialPoints['-Z']     = -automaticSpecialPoints['Z']

            automaticSpecialPoints['A']      = self._calcCircumcenter(2*automaticSpecialPoints['X'],2*automaticSpecialPoints['Y'])
            automaticSpecialPoints['-A']     = -automaticSpecialPoints['A']

            automaticSpecialPoints['B']      = self._calcCircumcenter(2*automaticSpecialPoints['Y'],2*automaticSpecialPoints['Z'])
            automaticSpecialPoints['-B']     = -automaticSpecialPoints['B']

            automaticSpecialPoints['C']      = self._calcCircumcenter(2*automaticSpecialPoints['Z'],2*automaticSpecialPoints['-X'])
            automaticSpecialPoints['-C']     = -automaticSpecialPoints['C']

        # === explicit lattice vector dependency? ===
        if self.getDimensionality() != 0:

            if self.getDimensionality() == 1:
                normal = self.__vecsReciprocal[0].copy()[::-1]
                normal[1] *= -1
                trafo = np.array([self.__vecsReciprocal[0],normal]).T

            if self.getDimensionality() == 2:
                trafo = np.array([self.__vecsReciprocal[0],self.__vecsReciprocal[1]]).T

            # get rid of the explicit lattice vector dependency
            if reciprocalBasis:
                trafo = np.linalg.inv(trafo)
                for k in iter(automaticSpecialPoints.keys()):
                    automaticSpecialPoints[k] = np.dot(trafo,automaticSpecialPoints[k])

            # introduce the explicit lattice vector dependency
            if not reciprocalBasis:
                for k in iter(userdefinedSpecialPoints.keys()):
                    userdefinedSpecialPoints[k] = np.dot(trafo,userdefinedSpecialPoints[k])

        for k in iter(userdefinedSpecialPoints.keys()):
            automaticSpecialPoints[k] = userdefinedSpecialPoints[k]

        return automaticSpecialPoints

    def addSpecialPoint(self,label,pos):
		#---Look at lines 105-107: the point you define is np.dot()'ed with the reciprocal lattice vectors. Define point therefore in units of the RLV's---#
        """Add a special point."""

        self.__specialPoints[label] = pos

    def addLatticevector(self,vector):
        """Add a lattice vector and calculate the reciprocal vectors."""

        # === add lattice vector ===
        # validation of vector shape
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Lattice vectors have to be 2D vectors.")

        # append vector to the array of lattice vectors
        if self.__vecsLattice.shape[0] == 0:
            self.__vecsLattice = np.array([vector], dtype=np.float)
        else:
            self.__vecsLattice = np.append(self.__vecsLattice,[vector], axis=0)

        # validation of lattice vector number
        if self.__vecsLattice.shape[0] > 2:
            raise Exception("There must be at most 2 lattice vectors.")

        self.__vecsReciprocal = self.getReciprocalVectors()

    def addBasisvector(self,vector):
        """Add a basis vector."""

        # === add basis vector ===
        # validation of vector shape
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Basis vectors have to be 2D vectors.")

        # append vector to the array of lattice vectors
        if self.__vecsBasis.shape[0] == 0:
            self.__vecsBasis = np.array([vector], dtype=np.float)
            self.__idxBasis = np.array([0])
            self.__idxSub = np.array([0])
        else:
            self.__vecsBasis = np.append(self.__vecsBasis,[vector], axis=0)
            self.__idxBasis = np.append(self.__idxBasis,self.__idxBasis[-1]+1)
            self.__idxSub = np.append(self.__idxSub,self.__idxSub[-1]+1)

    def _calcCircumcenter(self,vectorB, vectorC):
        """See http://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates."""

        D = 2*(vectorC[1]*vectorB[0]-vectorB[1]*vectorC[0])
        x = (vectorC[1]*np.vdot(vectorB,vectorB)-vectorB[1]*np.vdot(vectorC,vectorC))/D
        y = (vectorB[0]*np.vdot(vectorC,vectorC)-vectorC[0]*np.vdot(vectorB,vectorB))/D
        return np.array([x,y])

    def getKvectorsZone(self, resolution, dilation = True):
        """Calculate a matrix that contains all the kvectors of the Brillouin zone.

        kvectors = getKvectorsZone(resolution, dilation = True)
        kvectors[idxX, idxY, idxCoordinate]"""

        if self.__vecsReciprocal.shape[0] == 0:
            raise Exception("The 0D Brillouin zone is just a point. Use kvecs=None in System.solve instead.")

        elif self.__vecsReciprocal.shape[0] == 1:
            # === 1D Brillouin zone ===
            pos = self.__vecsReciprocal[0]/2
            positions = np.transpose([np.linspace(-pos[0],pos[0],resolution,endpoint=False),
                np.linspace(-pos[1],pos[1],resolution,endpoint=False)])

            step = positions[1]-positions[0]
            positions = np.array([positions[0]-step]+positions.tolist()+[positions[-1]+step])

            positionsMask = np.ones(positions.shape[:-1],dtype=np.bool)
            positionsMask[1:-1] = False

            positions = Kvectors(positions, mask = positionsMask)

        else:
            # === 2D Brillouin zone ===
            # reciprocal positions (contains the boundaries of the desired BZ)
            matTrafo = np.array([self.__vecsReciprocal[0], self.__vecsReciprocal[1]]).T

            reciprocalpositions     = np.empty((3*3,2))
            for n,[x,y] in enumerate(itertools.product([0,-1,1],[0,-1,1])):
                reciprocalpositions[n] = np.dot(matTrafo, [x,y])

            # calculate "radius" of the BZ (the resulting BZ will be too large; the areas which ware not relevant for the desired BZ will be masked)
            radius = np.max(np.sqrt(np.sum(self.__vecsReciprocal**2,axis=-1)))

            # generate a matrix [IdxX, IdxY, Coord] that stores the positions inside the too large BZ
            positions=np.mgrid[-radius:radius:2j*resolution,
                -radius:radius:2j*resolution,].transpose(1,2,0)

            # calculate the distances of the matrix points from the reciprocal positions of the desired BZ
            distances = np.tile(positions, (reciprocalpositions.shape[0],1,1,1))
            distances -= reciprocalpositions[:,None,None,:]
            distances = np.sqrt(np.sum(distances**2,axis=-1))

            # --- mask all points that are not close to the central position ---
            positionsMask = np.argmin(distances,axis=0) > 0

            # slice the matrices
            si, se = np.where(~positionsMask)
            slice = np.s_[si.min()-1:si.max() + 2, se.min()-1:se.max() + 2] # TODO why not "si.min():si.max() + 1, se.min():se.max() + 1"?

            positions = Kvectors(positions[slice], mask = positionsMask[slice])

        return positions

    def getKvectorsBox(self, resolution):

        if self.__vecsReciprocal.shape[0] == 0:
            # === 0D Brillouin box ===
            positions = Kvectors([[[0,0]]])

        elif self.__vecsReciprocal.shape[0] == 1:
            # === 1D Brillouin box ===
            l1 = np.linalg.norm(self.__vecsReciprocal[0])

            x,step = np.linspace(0, l1, resolution,endpoint=False,retstep=True)
            x = np.array([x[0]-step]+x.tolist()+[x[-1]+step])
            y = np.zeros_like(x)

            positions=np.transpose([x,y],(1,0))

            a = -np.arctan2(self.__vecsReciprocal[0,1],self.__vecsReciprocal[0,0])
            matRotate = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]).T
            positions = np.dot(positions,matRotate)

            positionsMask = np.ones(positions.shape[:-1],dtype=np.bool)
            positionsMask[1:-1] = False

            positions = Kvectors(positions, mask = positionsMask)

        else:
            # === 2D Brillouin box ===
            l1 = np.linalg.norm(self.__vecsReciprocal[0])
            l2 = np.linalg.norm(self.__vecsReciprocal[1])

            angle = np.abs(np.arccos(np.dot(self.__vecsReciprocal[0],self.__vecsReciprocal[1])/(l1*l2)))

            l2*=np.sin(angle)

            x,step = np.linspace(0, l1, resolution,endpoint=False,retstep=True)
            x = np.array([x[0]-step]+x.tolist()+[x[-1]+step])
            y,step = np.linspace(0, l2, resolution,endpoint=False,retstep=True)
            y = np.array([y[0]-step]+y.tolist()+[y[-1]+step])

            positions=np.transpose(np.meshgrid(x, y),(2,1,0))

            a = -np.arctan2(self.__vecsReciprocal[0,1],self.__vecsReciprocal[0,0])
            matRotate = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]).T
            positions = np.dot(positions,matRotate)

            positionsMask = np.ones(positions.shape[:-1],dtype=np.bool)
            positionsMask[1:-1,1:-1] = False

            positions = Kvectors(positions, mask = positionsMask)

        return positions

    def getKvectorsRhomboid(self, resolution):

        if self.__vecsReciprocal.shape[0] == 0:
            # 0d BZ
            positions = Kvectors([[[0,0]]])

        elif self.__vecsReciprocal.shape[0] == 1:
            # 1d BZ
            positions = self.__vecsReciprocal[0][None,:]*np.linspace(0, 1, resolution,endpoint=False)[:,None]
            positions = Kvectors(positions)

        else:
            # 2d BZ
            positions1 = self.__vecsReciprocal[0][None,:]*np.linspace(0, 1, resolution,endpoint=False)[:,None]
            positions2 = self.__vecsReciprocal[1][None,:]*np.linspace(0, 1, resolution,endpoint=False)[:,None]
            positions = positions2[:,None,:]+positions1[None,:,:]
            positions = Kvectors(positions)

        return positions

    def getKvectorsPath(self, resolution, pointlabels=None, points=None):
        """Calculate an array that contains the kvectors of a path through the Brillouin zone

        kvectors, length = getKvectorsPath(resolution, pointlabels=["G","X"])
        kvectors[idxPosition, idxCoordinate]"""

        if pointlabels is not None:
            specialPoints = self.getSpecialPoints()
            points = np.array([specialPoints[p] for p in pointlabels])
        elif points is not None:
            points = np.array(points)
        else:
            points = np.array(["G","G"])

        numPoints = points.shape[0]

        # path through the points
        stepsize = np.sum(np.sqrt(np.sum(np.diff(points,axis=0)**2,axis=-1)))/resolution

        positions = [None]*(numPoints-1)
        for n in range(1,numPoints):
            start = points[n-1]
            end = points[n]

            if stepsize == 0: steps = 1
            else: steps = max(int(np.round(np.linalg.norm(end-start)/stepsize)),1)

            newpos = np.transpose([np.linspace(start[0],end[0],steps,endpoint=False),
                np.linspace(start[1],end[1],steps,endpoint=False)])

            if n == 1: # first round
                step = newpos[1]-newpos[0]
                positions[n-1] = np.array([newpos[0]-step]+newpos.tolist())
            elif n == numPoints-1: # last round
                step = newpos[1]-newpos[0]
                positions[n-1] = np.array(newpos.tolist()+[newpos[-1]+step])
            else:
                positions[n-1] = newpos

        positions = np.vstack(positions)


        # save the labels and positions of special points
        pos = positions.copy()
        specialpoints_idx = []
        for p in points:
            idx = np.nanargmin(np.sum((pos-p)**2,axis=-1))
            specialpoints_idx.append(idx)
            pos[:,0][idx] = np.nan
            pos[:,1][idx] = np.nan

        specialpoints_labels = pointlabels

        # mask
        positionsMask = np.ones(positions.shape[:-1],dtype=np.bool)
        positionsMask[1:-1] = False

        return Kvectors(positions, positionsMask, specialpoints_idx, specialpoints_labels)

    def getPositions(self, cutoff):
        """Generate all positions from the lattice vectors using [0,0] as the basis vector.

        positions = getPositions(cutoff)
        positions[idxPosition, idxCoordinate]"""

        # value that is added to the cutoff to be on the save side
        numSubs = self.numSublattices()
        pos = np.tile(self.__vecsBasis, (numSubs,1,1))
        pos -= pos.transpose(1,0,2)
        safetyregion = np.max(np.sqrt(np.sum(pos**2,axis=-1)))

        # array that will contain all positions
        positions = []

        # --- first shift (do it only if two lattice vectors exist) ---
        shiftidx1 = 0
        boolLoop1 = True
        while boolLoop1:
            if self.__vecsLattice.shape[0] >= 2:
                shiftedpos1 = shiftidx1*self.__vecsLattice[1]

                # substract the other lattice vector to be as central as possible inside the cutoff region
                shiftedpos1 -= np.round(np.vdot(shiftedpos1,self.__vecsLattice[0])/
                    np.linalg.norm(self.__vecsLattice[0])**2 )*self.__vecsLattice[0]

                if shiftidx1 < 0: shiftidx1 -= 1
                else: shiftidx1 += 1

                # change looping direction / break if the shift is larger than a cutoff
                if np.linalg.norm(shiftedpos1) > cutoff+safetyregion:
                    if shiftidx1 > 0:
                        shiftidx1 = -1
                        continue
                    else:
                        break
            else:
                shiftedpos1 = np.array([0,0])
                boolLoop1 = False

            # --- second shift (do it only if at least one lattice vector exists) ---
            shiftidx0 = 0
            boolLoop0 = True
            while boolLoop0:
                if self.__vecsLattice.shape[0] >= 1:
                    shiftedpos0 = shiftidx0*self.__vecsLattice[0]

                    # add together all shifts
                    shiftedpos = shiftedpos1+shiftedpos0

                    if shiftidx0 < 0: shiftidx0 -= 1
                    else: shiftidx0 += 1

                    # change looping direction / break if the sum of shifts is larger than a cutoff
                    if np.linalg.norm(shiftedpos) > cutoff+safetyregion:
                        if shiftidx0 > 0:
                            shiftidx0 = -1
                            continue
                        else:
                            break
                else:
                    shiftedpos = np.array([0,0])
                    boolLoop0 = False

                # append the sum of shifts to the array of positions
                positions.append(shiftedpos)

        return np.array(positions,dtype=np.float)

    def getGeometry(self, cutoff):
        """Generate all positions from the lattice vectors using all the basis vectors.

        geometry = getGeometry(cutoff)
        geometry[idxSublattice, idxPosition, idxCoordinate]"""

        numSubs = self.numSublattices()

        # === creation of all positions ===
        positions = self.getPositions(cutoff)
        positionsAll = np.tile(positions, (numSubs,1,1)) + self.__vecsBasis[:,None]

        return positionsAll

    def getVecsLattice(self):
        """Get array of lattice vectors"""

        return self.__vecsLattice

    def getVecsBasis(self):
        """Get array of basis vectors"""

        return self.__vecsBasis

    def getIdxsBasis(self):
        """Get array of basis indices"""

        return self.__idxBasis

    def getIdxsSub(self):
        """Get array of sub lattice indices"""

        return self.__idxSub

    def getNumLattice(self):
        """Get length of array of lattice vectors"""

        return len(self.__vecsLattice)

    def getNumBasis(self):
        """Get length of array of basis vectors"""

        return len(self.__vecsBasis)

    def makeFiniteCircle(self, cutoff, center=[0,0]):
        """Generate a finite circular lattice.

        makeFiniteCircle(radius, center=[x,Y])"""

        numSubs = self.numSublattices()
        positions = self.getPositions(cutoff)
        positionsAll = (np.tile(positions, (numSubs,1,1)) + self.__vecsBasis[:,None]).reshape(-1,2)

        # save which sublattice corresponds to which position
        self.__idxSub = (np.zeros((numSubs,len(positions)),dtype=np.int) + self.__idxSub[:,None]).reshape(-1)

        # masking
        positionsAllAbs = np.sqrt(np.sum((positionsAll-center)**2,axis=-1))
        positionsAllMask = (positionsAllAbs > cutoff)
        positionsAll = positionsAll[~positionsAllMask]
        self.__idxSub = self.__idxSub[~positionsAllMask]

        # save the finite system as basisvectors
        self.__vecsLattice = np.array([])
        self.__vecsReciprocal = np.array([])
        self.__vecsBasis = positionsAll
        self.__idxBasis = np.arange(len(self.__vecsBasis))

    def makeFiniteRectangle(self, cutoffX, cutoffY, center=[0,0]):
        """Generate a finite rectangular lattice.

        makeFiniteRectangle(2*width, 2*height, center=[x,y])"""

        numSubs = self.numSublattices()
        positions = self.getPositions(np.sqrt(cutoffX**2+cutoffY**2))
        positionsAll = (np.tile(positions, (numSubs,1,1)) + self.__vecsBasis[:,None]).reshape(-1,2)

        # save which sublattice corresponds to which position
        self.__idxSub = (np.zeros((numSubs,len(positions)),dtype=np.int) + self.__idxSub[:,None]).reshape(-1)

        # masking
        positionsAllMask = (np.abs(positionsAll[:,0]-center[0]) > cutoffX) | \
            (np.abs(positionsAll[:,1]-center[1]) > cutoffY)
        positionsAll = positionsAll[~positionsAllMask]
        self.__idxSub = self.__idxSub[~positionsAllMask]

        # save the finite system as basisvectors
        self.__vecsLattice = np.array([])
        self.__vecsReciprocal = np.array([])
        self.__vecsBasis = positionsAll
        self.__idxBasis = np.arange(len(self.__vecsBasis))

    def makeFiniteAlongdirection(self, idxVecLattice, repetitions):
        """Make the basis finite in the direction of a lattice vector.

        makeFiniteAlongdirection(idxVecLattice, repetitions)"""

        numLatticevectors = self.__vecsLattice.shape[0]

        r = np.ones(numLatticevectors)
        r[idxVecLattice] = repetitions
        f = np.zeros(numLatticevectors, dtype=np.bool)
        f[idxVecLattice] = True

        self.enlargeBasis(r,f)

    def clipFiniteRectangle(self, cutoffX = np.inf, cutoffY = np.inf, center=[0,0]):
        """Clip basis in shape of a rectangle.

        clipFiniteRectangle(self, 2*width, 2*height, center=[x,y])"""

        # masking
        basisMask = (np.abs(self.__vecsBasis[:,0]-center[0]) > cutoffX) | \
            (np.abs(self.__vecsBasis[:,1]-center[1]) > cutoffY)
        self.__vecsBasis = self.__vecsBasis[~basisMask]
        self.__idxSub = self.__idxSub[~basisMask]

        self.__idxBasis = np.arange(len(self.__vecsBasis))

    def enlargeBasis(self, repetitions, makefinite=False):
        """Enlarge the basis (and make it finite if desired) in the direction of the lattice vectors.

        enlargeBasis(repetitions, makefinite)"""

        numLatticevectors = self.__vecsLattice.shape[0]

        if type(repetitions) is int: repetitions = np.ones(numLatticevectors)*repetitions
        if type(makefinite) is bool: makefinite = np.ones(numLatticevectors,dtype=np.bool)*makefinite

        # save new basis vectors
        for idxVecLattice, rep in enumerate(repetitions):
            numSubs = self.__vecsBasis.shape[0]

            positions = np.arange(rep)[:,None]*self.__vecsLattice[idxVecLattice][None,:]
            positionsAll = (np.tile(positions, (numSubs,1,1)) + self.__vecsBasis[:,None]).reshape(-1,2)
            self.__vecsBasis = positionsAll

            # save which sublattice corresponds to which position
            self.__idxSub = (np.zeros((numSubs,len(positions)),dtype=np.int) + self.__idxSub[:,None]).reshape(-1)

            # rescale lattice vectors
            self.__vecsLattice[idxVecLattice] *= rep

        self.__idxBasis = np.arange(len(self.__vecsBasis))

        # remove lattice vectors if desired
        boolarr = np.ones(self.__vecsLattice.shape[0],dtype=np.bool)
        boolarr[np.array(makefinite)] = False
        self.__vecsLattice = self.__vecsLattice[boolarr]

        # generate new reciprocal vectors
        self.__vecsReciprocal = self.getReciprocalVectors()

    def addRandomVacanciesByDensity(self, density, fixed = None):
        """Randomly remove basis vectors (useful for finite systems or large unit cells).

        The parameter `density` determines the density of vacancies. The parameter `fixed` specify a lattice site which must not be removed.
        """

        numLeft = int(len(self.__vecsBasis) * (1 - density))

        for n in range(10000):
            idxarray = np.arange(len(self.__vecsBasis))
            np.random.shuffle(idxarray)
            if fixed is None or fixed in self.__idxBasis[idxarray][:numLeft]: break
        else: raise Exception("Unable to remove lattice sites.")

        self.__vecsBasis = self.__vecsBasis[idxarray][:numLeft]
        self.__idxBasis = self.__idxBasis[idxarray][:numLeft]
        self.__idxSub = self.__idxSub[idxarray][:numLeft]

    def addRandomVacanciesByProbability(self, probability, fixed = None):
        """Randomly remove basis vectors (useful for finite systems or large unit cells).

        The parameter `density` determines the density of vacancies. The parameter `fixed` specify a lattice site which must not be removed.
        """

        for n in range(10000):
            boolarray = np.random.rand(self.numSublattices()) > probability
            if fixed is None or fixed in self.__idxBasis[boolarray]: break
        else: raise Exception("Unable to remove lattice sites.")

        self.__vecsBasis = self.__vecsBasis[boolarray]
        self.__idxBasis = self.__idxBasis[boolarray]
        self.__idxSub = self.__idxSub[boolarray]

    def addRandomShifts(self, standarddev):
        """Randomly shift lattice sites"""

        self.__vecsBasis += np.random.normal(scale=standarddev,size=self.__vecsBasis.shape)

        # bring basis vectors back into unit cell, the random shifts might have brought the basis vectors outside the cell

        for idx in range(len(self.__vecsBasis)):
            for vecLattice in self.__vecsLattice:

                # projection into the direction of the lattice vector
                proj = np.vdot(self.__vecsBasis[idx], vecLattice)/np.linalg.norm(vecLattice)

                # subtract lattice vectors
                self.__vecsBasis[idx] -= np.floor(proj/np.linalg.norm(vecLattice)) * vecLattice

    def numSublattices(self):
        """Returns the number of sublattices"""

        return self.__vecsBasis.shape[0]

    def getDimensionality(self):
        """Returns the number of lattice vectors (number of periodic directions)"""

        return self.__vecsLattice.shape[0]

    def getReciprocalVectors(self):
        """Returns the reciprocal lattice vectors (and saves them internally)."""

        dim = self.getDimensionality()

        if dim == 0:
            return np.array([])
        elif dim == 1:
            return np.array([
                2*np.pi*self.__vecsLattice[0]/np.linalg.norm(self.__vecsLattice[0])**2
                ])
        elif dim == 2:
            vecs = np.array([
                np.dot(np.array([[0,1],[-1,0]]),self.__vecsLattice[1]),
                np.dot(np.array([[0,-1],[1,0]]),self.__vecsLattice[0])
                ],dtype=np.float)
            vecs[0] = 2*np.pi*vecs[0]/ (np.vdot(self.__vecsLattice[0], vecs[0]))
            vecs[1] = 2*np.pi*vecs[1]/ (np.vdot(self.__vecsLattice[1], vecs[1]))
            return vecs
        else:
            raise Exception("Lattices with more than 2 lattice vectors are not supported")

    def plot(self, filename=None,show=True,cutoff=10):
        """Plot the lattice."""

        import matplotlib.pyplot as plt

        fig = plt.gcf()

        for p,b in zip(self.getGeometry(cutoff),self.__vecsBasis):
            line, = plt.plot(p[:,0],p[:,1], 'o',ms=4)
            fig.gca().add_artist(plt.Circle(b,cutoff, fill = False ,
                ec=line.get_color(),alpha=0.5,lw=1))
            plt.plot(b[0],b[1], 'kx',ms=7,mew=1)
        plt.axes().set_aspect('equal')
        plt.xlim(-1.5*cutoff,1.5*cutoff)
        plt.ylim(-1.5*cutoff,1.5*cutoff)

        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()

    def getDisplacements(self, cutoff):
        """Create a Displacements object that contains all vectors from the central position of a
        sublattice to all positions of another one."""

        # positions generated from the lattice vectors
        positions = self.getPositions(cutoff)
        sorter = np.argsort(np.sum(positions**2,axis=-1))
        positions = positions[sorter]

        # shifts given by the basisvectors
        shifts = self.__vecsBasis

        # === numbers ===
        # maximal number of links between the central position of a sublattice and all positions of another one
        numLinks = positions.shape[0]

        # number of sublattices
        numSubs = shifts.shape[0]

        # === creation of the distance matrix ===
        # array of central positions [Sub, Coord] that will be repeated to create the matrix matDeltaR
        positionsCentral = shifts

        # array of all positions [Sub, Link, Coord] that will be repeated to create the matrix matDeltaR
        positionsAll = np.tile(positions, (numSubs,1,1)) + positionsCentral[:,None]

        # creation of the matrix matDeltaR [Sub1, Sub2, Link, Coord]
        matPositionsCentral = np.tile(positionsCentral, (numLinks,numSubs, 1,1)).transpose(2,1,0,3)
        matPositionsAll = np.tile(positionsAll, (numSubs,1,1,1))
        matDeltaR = matPositionsAll-matPositionsCentral

        # masking of the matrix matDeltaR [Sub1, Sub2, Link, Coord]
        matDeltaRAbs = np.sqrt(np.sum(matDeltaR**2,axis=-1))
        matDeltaRMask = (matDeltaRAbs > cutoff) | (matDeltaRAbs < self.__tol)
        unnecessaryLinks = np.all(matDeltaRMask,axis=(0,1))

        return Displacements(matDeltaR[:, :, ~unnecessaryLinks],
                             positions[~unnecessaryLinks],
                             matDeltaRMask[:, :, ~unnecessaryLinks], self.__idxSub)

    def __str__(self):
        return str({'vecsLattice': self.__vecsLattice, 'vecsBasis': self.__vecsBasis})
