import sys
import numpy as np
sys.path.append("../")
from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import SquareLattice, TriangularLattice, HoneycombLattice
import DipolarStructure as ds

#Specify parameters, here tbar is intended to be the energy axis and is as such eliminated from iterations
lattice = HoneycombLattice()
cutoff = 2
tbar = 1.
t = np.array([0.54])/tbar
w = np.linspace(-5,5,50)/tbar
mu = np.linspace(-20,20,50)/tbar

#Packaging of parameters using dictionaries. For the units argument, the unit label and value come together in a list with the label as the first argument of the list.
units= {'energy':['tbar',tbar],'yaxis':['mu',mu],'xaxis':['w',w],'zaxis':['t',t]}
params = {'lattice':lattice,'cutoff':cutoff,'units':units}


"""=====================CHERN REGIONS AND BANDGAPS==============================
================================================================================
Arguments chern_labels and region_boundaries are optional arguments to plotImages,
but attempt labeling/outlining unique topological regions by chern numbers."""

#Run imaging to label for chern numbers
images = ds.getBandGapImages(params,resolution=64)
chern_labels, region_boundaries = ds.getChernRegions(images,params,resolution=64)

#plot with and without saving
ds.plotImages(images,units,chern_labels,region_boundaries,filename="HoneycombLattice_test.pdf")
ds.plotImages(images,units,chern_labels,region_boundaries)




"""=====================FLATNESS AND BANDGAPS===================================
================================================================================
Use argument getFlatness=True, returns flatness images over the parameter space 
given by params. Flattness is plotted also using plotImages. By changing the band
parameter in ds.getBandGapImages(), one can investigate different bands over the
parameter space. For band=None, all bands are considered; however, only the minimum
bandgap and flatness are saved for the image."""

#Run imaging for lowest band obtaining in addition the flatness
bandgap_images, bandflatness_images = ds.getBandGapImages(params,getFlatness=True,band=0)  

ds.plotImages(bandgap_images, units, filename="single_particle_lowestband_bandgap_cutoff_2.png")
ds.plotImages(bandflatness_images, units, cbar_label="Flatness", filename="single_particle_lowestband_flatness_cutoff_2.png")
