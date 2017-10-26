# -*- coding: utf-8 -*-
"""
This script calculates the average of .nii warped red images

@author: Dreosti Lab
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

Fish_1 = 'F:/Hande_Confocal/Analysis/Controls/C_Fos_Control_Fish_N2_20dpf_2017_07_05/C_2_Ref_warped_red.nii.gz'
Fish_2 = 'F:/Hande_Confocal/Analysis/Controls/C_Fos_Control_Fish_N1_20dpf_2017_07_05/C_1_Ref_warped_red.nii.gz'

Image_1 = nib.load(Fish_1)
Image_2 = nib.load(Fish_2)

Image_1_size = Image_1.shape
Image_1_type = Image_1.get_data_dtype()

Image_2_size = Image_1.shape
Image_2_type = Image_1.get_data_dtype()

Image_data_1 = Image_1.get_data()
Image_data_2 = Image_2.get_data()

z_stack_1 = np.mean(Image_data_1,axis=2)
z_stack_1_max = np.max(Image_data_1,axis=2)
z_stack_1_std = np.std(Image_data_1,axis=2)

plt.figure()
plt.imshow(z_stack_1_max)

plt.figure()
plt.imshow(z_stack_1_std)

plt.figure()
plt.imshow(z_stack_1)

Average_images = (Image_data_1+Image_data_2)/2

plt.figure()
for i in range(0,Image_1_size[2]):
    plt.cla()
    plt.imshow(Average_images[:,:,i])
    plt.show()
    plt.pause(0.1)


new_img = nib.Nifti1Image(Average_images, Image_1.affine, Image_1.header)
nib.save(new_img, "Average.nii.gz")

# FIN
