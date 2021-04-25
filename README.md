# Image Blending
A program for blending two images using a pyramid blending
![alt text](https://github.com/IdoSagiv/image-blending/blob/main/images/blended_image.png?raw=true)
## Technologies And Tools
This program is written in Python and using the libraries: 'numpy', 'scipy', 'imageio', 'skimage' and 'mayplotlib'.<br/>
The program was developed on Linux, and tested both on Linux and Windows.
## Overview
The program gets as input three images - two input images to blend and one mask image.<br/>
The blend is done using a Laplacian pyramids of the two input images and a Gaussian pyramid of the mask image.<br/>
After building the described pyramids, the program creates a new, third, Laplacian pyramid in which every level is the 
sum of the corresponding layers in the two Laplacian pyramids weighted by the corresponds layer in the mask image Gaussian pyramid<br/>
i.e if the input images Laplacian pyramids are ![](https://latex.codecogs.com/svg.latex?\Large&space;L_A) and ![](https://latex.codecogs.com/svg.latex?\Large&space;L_B) 
and the mask Gaussian pyramid is ![](https://latex.codecogs.com/svg.latex?\Large&space;G_M) then on every layer of the output image Laplacian pyramid  ![](https://latex.codecogs.com/svg.latex?\Large&space;L_C) 
we get ![](https://latex.codecogs.com/svg.latex?\Large&space;L_C(i,j)=G_M(i,j)L_A(i,j)+(1-G_M(i,j))L_B(i,j))
## Blending Process
### Step 1 - Build a Gaussian pyramid to the two input images
![alt text](https://github.com/IdoSagiv/image-blending/blob/main/images/manhattan_gaussian.png?raw=true)<br/>
![alt text](https://github.com/IdoSagiv/image-blending/blob/main/images/space_gaussian.png?raw=true)
### Step 2 - Build a Laplacian pyramid to the two input images
Using the Gaussian pyramids from the first step ![](https://latex.codecogs.com/svg.latex?\Large&space;G_A) and ![](https://latex.codecogs.com/svg.latex?\Large&space;G_B), build a Laplacian pyramid to each of the input images.<br/>
Every layer ![](https://latex.codecogs.com/svg.latex?\Large&space;i<n) of the Laplacian pyramid ![](https://latex.codecogs.com/svg.latex?\Large&space;L) is defined by ![](https://latex.codecogs.com/svg.latex?\Large&space;L_i=G_i-expand(G\textsubscript{i+1})) and ![](https://latex.codecogs.com/svg.latex?\Large&space;L_n=G_n).<br/>while expand() is being done by zero padding + gaussian blur<br/>
![alt text](https://github.com/IdoSagiv/image-blending/blob/main/images/manhattan_laplacian.png?raw=true)<br/>
![alt text](https://github.com/IdoSagiv/image-blending/blob/main/images/space_laplacian.png?raw=true)
### Step 3 - Build a Gaussian pyramid to the mask image
![alt text](https://github.com/IdoSagiv/image-blending/blob/main/images/mask_gaussian.png?raw=true)
### Step 4 - Construct the blended image
Build the blended image Laplacian pyramid as described before, and construct the output image from it.<br/>
![alt text](https://github.com/IdoSagiv/image-blending/blob/main/images/all_images.png?raw=true)
### RGB images
In case the input images are colored, perform the described steps on every channel (R,G,B), and combine the three blended images.
