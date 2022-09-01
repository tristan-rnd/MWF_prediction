#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser(description='Slice a 3D nifti image into 2D tiff images.')
parser.add_argument('P', metavar='<file-path>', type=str,
                    help='Filepath of the image.')
parser.add_argument('-ax', metavar='<axis>', type=int, default=0,
                    help='Axis on which applying the slicing (x=0, y=1, z=2), default : 0.')
parser.add_argument('-f', metavar='<output-folder>', type=str, default="",
                    help='Relative folder for the output.')

import nibabel as nib
import numpy as np
from skimage.io import imsave
import warnings
warnings.filterwarnings('ignore','.*low contrast.*', )

args = parser.parse_args()

img = nib.load(args.P)
img = img.get_fdata()
axs = args.ax

if axs==0:
    for k in range(0,np.shape(img)[0]):
        name = args.P.rpartition('/')[2]
        name = name.partition('.')[0]
        name = args.f+name+"_slice_"+str(k)+".tiff"
        imsave(name,(img[k,:,:]))
elif axs==1:
    for k in range(0,np.shape(img)[1]):
        name = args.P.rpartition('/')[2]
        name = name.partition('.')[0]
        name = args.f+name+"slice_"+str(k)+".tiff"
        imsave(name,(img[:,k,:]))
elif axs==2:
    for k in range(0,np.shape(img)[2]):
        name = args.P.rpartition('/')[2]
        name = name.partition('.')[0]
        name = args.f+name+"slice_"+str(k)+".tiff"
        imsave(name,(img[:,:,k]))
else:
    print("Wrong axis")


