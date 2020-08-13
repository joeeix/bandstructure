#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The purpose of this library is to examine the topological bandstructure of general systems with dipolar interactions, as a function of the parameters within the respective Hamiltonian. The typical use is to first call getBandGapImages(), feeding the output to getChernRegions(), and then feeding outputs from getBandGapImages() and getChernRegions() to plotImages() for a visualization. Additionally, the arguments 'getFlatness' and 'band' in getBandGapImages() provide a means to investigate the bandgap or flatness of a specific band or bands. Important to the function is the packaging of the params argument, and unpackaging the output, see the example Dipolar_Band_Images.py."""

#standard imports
import sys 
import numpy as np
sys.path.append("../")

#imports within bandstructure
from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import *

#imports for region finding
from skimage.measure import label, regionprops
from skimage.filters import threshold_triangle #threshold_otsu
from skimage.segmentation import find_boundaries

#For iterations and parallel processing
from itertools import product
import multiprocessing as mp
from functools import partial

#plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as grdsp
from matplotlib.cm import ScalarMappable as scm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap


def getChernRegions(images,params,resolution=32):
    """Finds regions in a 2D image and obtains centroid. Then for each region, translates centroid
    coordinates into the parameter space (x,y), and calculates the chern number at that
    point. Returns array with dimension (Image number,Region number) which contains chern numbers and locations (x,y). Also returns array of region boundaries with dimension numImages."""
    
    
    units = params['units']
    enLabel, en = units['energy']
    xLabel, x = units['xaxis']
    yLabel, y = units['yaxis']
    zLabel, z = units['zaxis']
    names = [enLabel,xLabel,yLabel,zLabel]
    
    #for translating centroid coordinates of a region in an array denoted (i,j) into (x,y) space
    rangex, rangey = np.abs(x[0]-x[-1]),np.abs(y[0]-y[-1])
    minx, miny = np.min(x), np.min(y)
    
    #setup for parallelization and output
    iterFunc = partial(getCherns,params['lattice'],params['cutoff'],names,resolution,)
    chern_images = []
    boundaries = []
    
    #for each image, find regions and label using triangle thresholding method and skimage labeling. Note triangle thresholding is good for highly peaked structures, it is possible that Otsu's method (threshold_otsu) could be better depending on one's needs.
    for idx_img,image in enumerate(images):
        z_img = z[idx_img] 
        
        #creating labeled regions of an image from skimage thresholding
        thresh = threshold_triangle(image)
        bw = image > thresh
        label_image = label(bw)
        regions = regionprops(label_image)
        
        #finding region centroids, translating into coordinate space (x,y)
        image_chern_data = getCentroids(regions,x,y,z_img,en)
            
        #Calculating chern numbers for regions in parallel
        image_pool = mp.Pool(None)
        image_chern_list = image_pool.map(iterFunc,image_chern_data)
        image_pool.close()
        
        #Setup for while loop which finds subregions within regions found above
        converged = False
        num_regions = len(regions)+1
        iter_labels = label_image
        
        #To compile all unique regions in a single image and finding borders
        all_labeled_regions = label_image
        label_idx = num_regions
        
        #Iteratively thresholding regions to find subregions until converged, meaning no unique chern band structures were found 
        while not converged:
            #a mask to be filled with True in newly found regions
            new_bw = np.full(iter_labels.shape,False)
            
            #splitting each region into subregions, including the 0 region (background) in label_image, upon each iteration
            for idx_reg in range(num_regions):
                iter_chern_data = []
                
                sub_labels = getSubRegions(image,iter_labels,idx_reg)
                sub_regions = regionprops(sub_labels)
                
                if len(sub_regions) >= 1:
                    
                    iter_chern_data = getCentroids(sub_regions,x,y,z_img,en)
                    iter_pool = mp.Pool(None)
                    iter_chern_list = iter_pool.map(iterFunc,iter_chern_data)
                    iter_pool.close()
                    
                    #checking if any of the chern bands are not identical to ones previously found in the image
                    new_cherns = isin(iter_chern_list,image_chern_list,invert=True)
                    
                else:
                    
                    new_cherns = [False]
                    
                if any(new_cherns):
                    #keeping track that the iteration hasn't converged, adjusted at end of loop
                    converged += 1
                    
                    #taking the indices for sub regions with unique chern bands, and those with new chern bands. Index correspondes to the sub region label+1.
                    unique_indices = np.unique(iter_chern_list,return_index=True,axis=0)[1] 
                    new_indices = [i for i,x in enumerate(new_cherns) if x == True]
                    
                    #for subregions which have unique (to the region) and new (to the image) chern bands
                    for idx in np.intersect1d(new_indices,unique_indices):
                        #keep subregions with new found chern bands, to subdivide further until converged
                        new_bw += np.where(sub_labels==idx+1,True,False)
                        
                        #update labeled image with new labels
                        all_labeled_regions[sub_regions[idx].coords[:,0],sub_regions[idx].coords[:,1]] = label_idx
                        label_idx+=1
                        
                        #add coordinates and chern numbers to the image list
                        image_chern_data.append(iter_chern_data[idx])
                        image_chern_list.append(iter_chern_list[idx])
                
                #end for idx_reg
            
            #Subregions with new chern bands are selected to be re-iterated over, and further subdivided
            iter_labels = label(new_bw)
            num_regions = len(regionprops(iter_labels))+1
            converged = not bool(converged)
            
            #end while not converged
            
        #Rearrange chern_list and coords to make readable chern labels for plotImages
        image_chern_data = np.array(image_chern_data)[:,1:3].tolist()
        chern_image = []
        
        for i in range(len(image_chern_list)):
            chern_image.append([image_chern_list[i],image_chern_data[i]])
        
        image_boundaries = find_boundaries(all_labeled_regions,mode='subpixel').astype(np.uint8)
        chern_images.append(chern_image)
        boundaries.append(image_boundaries)
        #end for image
        
    return chern_images, boundaries


def getCentroids(regions,x,y,z,en):
    """Takes a regionprops(label_image) object from skimage. Returns list of centroid coordinates in real coordinate space (x,y,z,en) for each labeled region.
    Helper function for getChernRegions."""
    
    #for translating centroid coordinates of a region in an array denoted (i,j) into (x,y) space
    rangex, rangey = np.abs(x[0]-x[-1]),np.abs(y[0]-y[-1])
    minx, miny = np.min(x), np.min(y)
    
    centroid_coords = []
    for reg_idx, area in enumerate(regions):
        data = [en]
        cx,cy = area.centroid
        cx,cy = (cx/len(x))*rangex+minx, (cy/len(y))*rangey+miny   #coords in real (x,y) units range
        data.append(cx)
        data.append(cy)
        data.append(z)
        centroid_coords.append(data)
    
    return centroid_coords


def getSubRegions(image,labeled_image,region_idx):
    """Helper Function for getChernRegions used to successively find unique chern substructures. Returns a masked region with subregion labels."""
    
    regions = regionprops(labeled_image)
    
    if region_idx > len(regions)+1 or region_idx < 0:
        raise Exception('region_idx does not exist in image, it was: '+str(region_idx)+' and the number of regions was: '+str(len(regions)+1))
    
    #Colors is the masked region in the image. If there is only 1 color (i.e. 1 bandgap value or 1 data point), thresholding fails. 
    mask = np.where(labeled_image == region_idx,True,False)
    masked_image = image*mask
    colors = np.ravel(image)[np.ravel(mask)]
    unique_colors = np.unique(colors)
    
    if len(unique_colors) > 1:
        #consider here different thresholding methods, e.g. otsu's method
        region_threshold = threshold_triangle(colors)
    else:
        region_threshold = 0
    
    masked_bw = masked_image > region_threshold
    masked_labeled_region = label(masked_bw)
    
    return(masked_labeled_region)


def getCherns(lattice,cutoff,names,resolution,data):
    """Returns list of chern numbers for all bands"""
    
    params = {'lattice':lattice,'cutoff':cutoff,names[0]:data[0],names[1]:data[1],names[2]:data[2],names[3]:data[3]}
    packedParams = Parameters(params)
    s = DipolarSystem(packedParams)
    Zone = lattice.getKvectorsRhomboid(resolution=resolution)
    BZ = s.solve(Zone,processes=1)
    nb = BZ.numBands()
    cherns = []
    if nb > 1:
        for band in range(nb):
            cherns.append((1./(2.*np.pi))*BZ.getBerryFlux(band))
    else:
        cherns.append([])
    
    return cherns

    
def isin(element,test,invert=False):
    """Modified isin function because np.isin checks each element of a row, not the row itself. This is to check if chern numbers of specific band structures have occurred. Element is checked against test. Returns an array with shape (len(element),) which tells if element[i] appeared in test."""
    
    element = np.array(element)
    test = np.array(test)
    result = np.full(element.shape[0],False)
    for i in test:
        result += np.equal(element,i).all(axis=1)
    
    if invert:
        return np.invert(result)
    
    else:
        return result


def plotImages(images,units,chern_labels=None,region_boundaries=None,filename=None,show=True,cbar_label="BandGap"):
    """Plotting images of minimum bandgap using getBandGapImages and optional labels for unique regions based on chern bands."""
    
    enLabel, en = units['energy']
    xLabel, x = units['xaxis']
    yLabel, y = units['yaxis']
    zLabel, z = units['zaxis']
    edges = [y[0],y[-1],x[-1],x[0]]
    
    #setup the figure so that each bandgap image is alotted a size (10,5) on the image grid; a separate grid area to the right of the figure for the colorbar
    fig = plt.figure(figsize=(10,len(images)*5))
    full_grid = grdsp.GridSpec(1,2,wspace=0.05,hspace=0.0,width_ratios=[0.8,0.2])
    image_grid = grdsp.GridSpecFromSubplotSpec(len(images),1,subplot_spec=full_grid[0,0],wspace=0.0,hspace=0.55)
    
    #coloring information
    cmap = 'viridis'
    vmin, vmax = np.min(images),np.max(images)
    norm = Normalize(vmin=vmin,vmax=vmax)
    
    #each axis is created on the image grid and the bandgap images are plotted
    for i,image in enumerate(images):
        #note x,y labels are reversed due to how matrices are handled
        ax = fig.add_subplot(image_grid[i])
        ax.set_xlabel(yLabel,fontsize=10) 
        ax.set_ylabel(xLabel,fontsize=10)
        ax.set_title(zLabel + "= " + str(z[i]),fontsize=10)
        ax.imshow(image,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='none',extent=edges)
        
        
    axes = fig.get_axes()
    
    #label the regions using the chern_labels, unpacking the labels 
    if np.any(chern_labels != None):
        
        for i,img in enumerate(chern_labels):
            
            for j,reg in enumerate(img):
                label = str(reg[0][:len(reg[0])//2]).strip(']')+'\n'+str(reg[0][len(reg[0])//2:]).strip('[')
                axes[i].text(reg[1][1],reg[1][0],label,fontsize=8)
    
    #overlay the boundaries on the images
    if np.any(region_boundaries != None):
        #create custom colormap for transparent red boundary lines. Colors values are (R,G,B,A). Only two points needed for 0=no boundary and 1=boundary.
        reds = [(1,0,0,0),(1,0,0,0.3)]
        boundary_cmap = LinearSegmentedColormap.from_list('bound_cmap',reds)
        
        #draw boundaries on each image
        for i in range(len(images)):
            axes[i].imshow(region_boundaries[i],cmap=boundary_cmap,interpolation='none',extent=edges)
    
    #make the colorbar #TODO for many images, i.e. many z, the colorbar is obnoxiously long consider rework
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(scm(norm=norm,cmap=cmap), cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=10)
    plt.tight_layout()
        
    #Output
    if filename is not None:
        plt.savefig(r"images/"+filename)

    if show:
        plt.show()
        

def getBandGapImages(params,resolution=32,getFlatness=False,band=None,processes=None):
    """Returns an array of image matrices (x,y) of Band Gaps and/or flatnesses for each z value. For band=None, the minimum bandgap/flatness of the entire bandstructure is returned.
    Image matrices have coordinates ~image[x,y] with x (down) and y (right) in the plotted image."""
    
    units = params['units']
    enLabel, en = units['energy']
    xLabel, x = units['xaxis']
    yLabel, y = units['yaxis']
    zLabel, z = units['zaxis']
    numImages = len(z)

    xyz = np.roll(np.array(list(product(z,x,y))),2,1)
    energy = en*np.ones(len(xyz))
    
    data = np.c_[energy,xyz]
    names = [enLabel,xLabel,yLabel,zLabel]
    
    iterFunc = partial(getBG,params['lattice'],params['cutoff'],names,resolution,getFlatness=getFlatness,band=band)
    
    #iterFunc = partial(getBG,params['lattice'],params['cutoff'],names,resolution,)
    
    pool = mp.Pool(processes)
    
    if getFlatness:
        band_output = pool.map(iterFunc,data)
        band_output = np.array(band_output)
        pool.close()
        
        bandGaps, bandFlats = np.array(band_output[:,0].tolist()), np.array(band_output[:,1].tolist()) #np.array(array.tolist()) fixes the data type to np.float instead of np.object
        
        #take only minimum flatness if more than 1 band
        if bandFlats.ndim > 1:
            bandFlats = np.amin(bandFlats,axis=1)
            
        return bandGaps.reshape(numImages,len(x),len(y)), bandFlats.reshape(numImages,len(x),len(y))
        
    else:
        bandGaps = pool.map(iterFunc,data)
        bandGaps = np.array(bandGaps)
        pool.close()
        return bandGaps.reshape(numImages,len(x),len(y))


def getBG(lattice,cutoff,names,resolution,data,getFlatness=False,band=None):
    """Returns minimum band gap in absolute value. Used by getBandGapImages."""
    
    params = {'lattice':lattice,'cutoff':cutoff,names[0]:data[0],names[1]:data[1],names[2]:data[2],names[3]:data[3]}
    packedParams = Parameters(params)
    s = DipolarSystem(packedParams)
    Zone = lattice.getKvectorsRhomboid(resolution=resolution)
    BZ = s.solve(Zone,processes=1)
    nb = BZ.numBands()
    if nb > 1:
        BandGap = np.amin(np.absolute(BZ.getGap(band=band)))
        if getFlatness:
            Flatness = BZ.getFlatness(band=band)
    else: 
        BandGap = 0.
        if getFlatness:
            Flatness = 0.
    
    if getFlatness:
        return BandGap, Flatness
    
    else:
        return BandGap






def getBoundaries(image):
    """This is a testing area to try and improve the current region finding scheme used by getChernRegions. Currently, regions are found by 
    iteratively thresholding regions with skimage. The idea is to instead find where a bandgap closes in the image, and take these as boundaries
    which distinctly outline and define the regions of interest. So far tried: derivatives/local minima, scipy.signal.argrelextrema; not as
    effective as the iterative method. Other idea: look directly at Hamiltonian/band structure and mathematically determine the criteria for a
    closed band gap."""








