#!/usr/bin/env python

import os
import pandas as pd

full_path_cluster = "/users/cn/lsantus/data/CrossEvoPred/berthelot"
full_path = "/home/luisasantus/Desktop/crg_cluster/data/CrossEvoPred/berthelot"
folders_list = ["Homo_sapiens", "Macaca_mulatta"]

# store fastqs in in a dataframe

# Dictionary to store the list of fastq.gz files in each folder
fastq = {}

for folder in folders_list:
    folder_path = os.path.join(full_path, folder)
    print(folder_path)  
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        fastq[folder] = [file for file in files if file.endswith(".fastq.gz")]






