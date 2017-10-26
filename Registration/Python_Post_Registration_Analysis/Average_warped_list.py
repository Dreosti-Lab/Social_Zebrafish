# -*- coding: utf-8 -*-
"""
This script calculates the average of .nii warped red images

@author: Dreosti Lab
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


folder_path = 'G:/Hande_Confocal/Analysis/'
#folder_list = 'F:/Hande_Confocal/Analysis/Python_Analysis/Controls.txt'

#folder_list = 'G:/Hande_Confocal/Analysis/Python_Analysis/Non_Social.txt'
#folder_list = 'F:/Hande_Confocal/Analysis/Python_Analysis/Social.txt'
folder_list = 'G:/Hande_Confocal/Analysis/Python_Analysis/Anti_Social.txt'

folder_file = open(folder_list, "r") #"r" means read the file
file_list = folder_file.readlines() # returns a list containing the lines

num_fish = len(file_list) 
Average_images = np.zeros((512, 512, 31), dtype = np.float32)
Divisor = np.zeros((512, 512, 31), dtype = np.float32)
Valid = np.zeros((512, 512, 31), dtype = np.float32)

for f in file_list:
    img_file = folder_path + f[:-1]  # to remove the space(new line character) in the txt file 

    Image = nib.load(img_file)
    
    Image_size = Image.shape
    Image_type = Image.get_data_dtype()
    Image_data = Image.get_data()
   
#    
#    z_stack = np.mean(Image_data,axis=2)
#    z_stack_max = np.max(Image_data,axis=2)
#    z_stack_std = np.std(Image_data,axis=2)
#    
#    plt.figure()
#    plt.imshow(z_stack_max)   
#    plt.figure()
#    plt.imshow(z_stack_std)  
#    plt.figure()
#    plt.imshow(z_stack)
    Valid = (Image_data != 0.0).astype(int)
    
    Divisor = Divisor + Valid
    
    Average_images = Average_images + Image_data

Average_images = Average_images/Divisor

#plt.figure()
#for i in range(0,Image_1_size[2]):
#    plt.cla()
#    plt.imshow(Average_images[:,:,i])
#    plt.show()
#    plt.pause(0.1)


new_img = nib.Nifti1Image(Average_images, Image.affine, Image.header)
nib.save(new_img, "Test.nii.gz")

# FIN
