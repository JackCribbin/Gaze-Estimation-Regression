# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:32:42 2023

@author: jackp
"""

import os
import random

count = 0
source = 'c:\\users\\jackp\\onedrive\\documents\\_regression\\TrainingImages'
destination = 'c:\\users\\jackp\\onedrive\\documents\\_regression\\TestingImages'

# Set the percentage of total training images to move to the testing folder 
rate = 0.05

# Gather all files
allfiles = os.listdir(source)
 
# Iterate on all files to move some of them to the destination folder
for f in allfiles:
    if(random.random() <= rate):
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        os.rename(src_path, dst_path)
        count = count + 1

print(count,'test files moved')