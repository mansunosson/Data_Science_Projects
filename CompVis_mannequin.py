# Script to convert front and side view image to mannequin

# NOTES:
# USE THE FACT that while arms are thinner than torso (side view), 
# they are cross sectionally spherical! I.e. width = min(width_sideview, width_frontview) 
# in other words, either the dimension is available from image, or the "hidden" dimension is identical to the one available 


#pip install numpy-stl
#pip install opencv-python
#from PIL import Image, ImageFilter

import time
import numpy as np
from stl import mesh
import os
import cv2
import matplotlib.pyplot as plt

os.chdir(r"path")
print(os.getcwd()) # Prints the current working directory


# Define function to convert to gray-scale
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = np.uint8(0.2989 * r + 0.5870 * g + 0.1140 * b)

    return gray

# Define function to get rid of texture and give body outline
def outline(im, c):
    #convert to grayscale
    im_g = rgb2gray(im)
    #resize image to standardized height
    AR = np.size(im_g,1)/np.size(im_g,0)
    im_g = cv2.resize(im_g, (np.uint8(500*AR), 500))
    #remove textures
    avg  = np.mean(im_g)
    im_g[im_g<avg*c] = 0
    im_g[im_g>avg*c] = 255   
    #clever use of blurring to get rid of hanging pixels
    #im_g = cv2.blur(im_g, (2,2), cv2.BORDER_DEFAULT)
    #im_g[(50 < im_g) & (im_g < 255)] = 255
    #im_g = cv2.blur(im_g, (2,2), cv2.BORDER_DEFAULT)
    #im_g[(50 < im_g) & (im_g < 255)] = 255    
    return im_g

# Define horizontal image gradient function
def imgrad(im):
    h = np.size(im,0)
    w = np.size(im,1)
    GH_im = np.absolute(im[:,0:(w-1)].astype(int)-im[:,1:(w)].astype(int))
    GV_im = np.absolute(im[0:h-1,:].astype(int)-im[1:h,:].astype(int))
    gradient = np.maximum(GH_im[1:h-1,1:w-1], GV_im[1:h-1,1:w-1])
    return gradient

# Define function to draw scatter plots of cross sections
def crossplt(dim, sec, IND):
    odim = np.array([0, 1, 2])
    odim = odim[odim != dim]
    if isinstance(sec, int):
        cs = IND[IND[:,dim] == sec]
        plt.scatter(cs[:,odim[0]], cs[:,odim[1]])
    else:
        for i in range(len(sec)):
            cs = IND[IND[:,dim] == i]
            plt.scatter(cs[:,odim[0]], cs[:,odim[1]])
            time.sleep(0.1)
            
            
# Load front and side images, convert to outline and calculate gradients
side_im  = cv2.imread('side.jpg')
front_im = cv2.imread('front.jpg')

# Show front and side image
plt.figure()
plt.imshow(front_im)
plt.show()

plt.figure()
plt.imshow(side_im)
plt.show()

# Resize, get outlines and gradients of front and side images
front_im   = outline(front_im,1.1)
side_im    = outline(side_im,1.1)
front_grad = imgrad(front_im)
side_grad  = imgrad(side_im)
plt.figure()
plt.imshow(front_grad)
plt.show()
plt.figure()
plt.imshow(side_grad)
plt.show()


# Convert gradient to points in 3D space
IND_f = np.zeros((np.size(front_grad,0), 3), dtype=int)
IND_s = np.zeros((np.size(side_grad, 0), 3), dtype=int)
points_f = np.nonzero(front_grad)
points_s = np.nonzero(side_grad)
IND = np.zeros((np.size(points_f,1)+np.size(points_s,1), 3), dtype=int)

# height (X)
IND[range(np.size(points_f,1)),0] = points_f[0]
IND[range(np.size(points_f,1),np.size(points_f,1)+np.size(points_s,1)),0] = points_s[0]

# width (Y)
IND[range(np.size(points_f,1)),1] = points_f[1]-np.mean(points_f[1]) 

# depth (Z)
IND[range(np.size(points_f,1),np.size(points_f,1)+np.size(points_s,1)),2] = points_s[1]-np.mean(points_s[1]) 

# Draw scatter plots of cross sections         
for i in range(10,400):
    plt.figure()
    plt.xlim(-60, 60)
    plt.ylim(-45,45)
    crossplt(0, i, IND)
    plt.show()

# Write function to draw ellipses by scanning the height dimension and skip points that are disjoint (dont have neighbours above and below). Draw several ellipses
# if there are smooth transitions in the X-dimension 



