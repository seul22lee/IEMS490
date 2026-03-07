# A GAMMA wrapper for running MPC
# import for GAMMA =================================================================
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import cupy

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

# import torch
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import warnings
import subprocess
warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


import matplotlib.pyplot as plt
import numpy as np
import zarr
import pandas as pd
from tqdm import tqdm
from gamma_model_simulator import GammaModelSimulator
import os

import os
import numpy as np
import cupy as cp
import gamma.interface as rs
from multiprocessing import Process
import time

#num = 0
#os.environ['CUDA_VISIBLE_DEVICES'] = f'{num}'


import os
import subprocess
import shutil
from scipy.interpolate import Rbf

from get_melt_pool_temp_width_depth import get_meltpool_temp_width_depth


class GAMMA_obj():
    def __init__(self, INPUT_DATA_DIR, SIM_DIR_NAME, BASE_LASER_FILE_DIR, CLOUD_TARGET_BASE_PATH, solidus_temp, window, init_runs, sim_interval, laser_power_number):
        self.INPUT_DATA_DIR = INPUT_DATA_DIR
        self.SIM_DIR_NAME = SIM_DIR_NAME
        self.BASE_LASER_FILE_DIR = BASE_LASER_FILE_DIR
        self.CLOUD_TARGET_BASE_PATH = CLOUD_TARGET_BASE_PATH

        self.laser_profile_number = laser_power_number
        self.laser_profile_filename_simulation = f"laser_profile_{laser_power_number}"
        self.laser_profile_filename = f"{self.laser_profile_filename_simulation}.zarr"
        self.LASER_FILE = os.path.join(BASE_LASER_FILE_DIR, self.laser_profile_filename_simulation)

        self.solidus_temp = solidus_temp
        self.window = window
        self.sim_interval = sim_interval
        self.initial_average_temp = None
        self.initial_average_depth = None
        
        self.GAMMA = rs.FeaModel(
                    input_data_dir=self.INPUT_DATA_DIR,
                    geom_dir=self.SIM_DIR_NAME,
                    laserpowerfile=self.LASER_FILE,
                    VtkOutputStep=1.,
                    ZarrOutputStep=0.02,
                    outputVtkFiles=False,
                    verbose=False
                    )
        
        self.melt_pool_temp_save = pd.DataFrame({"timestep":[], "temp":[]})
        self.melt_pool_width_save = pd.DataFrame({"timestep":[], "width":[]})
        self.melt_pool_depth_save = pd.DataFrame({"timestep":[], "depth":[]})
        self.global_counter = 0
        self.init_runs = init_runs
        self.non_zero_depth_buffer = None
        
    def run_initial_steps(self,laser_power = None):
        if self.init_runs == None:
            sim_steps = self.window*self.sim_interval
        else:
            sim_steps = self.init_runs*self.sim_interval
            
        for i in tqdm(range(sim_steps)):
            self.GAMMA.run_onestep(None)
            laser_on = 1
            self.melt_pool_temp_save, self.melt_pool_width_save, self.melt_pool_depth_save = get_melt_pool(laser_on, self.GAMMA, self.global_counter, self.melt_pool_temp_save, self.melt_pool_width_save, self.melt_pool_depth_save, self.solidus_temp)
            self.global_counter += 1
            # output temperature averaging every sim_interval steps
            # Loop through the sequence in steps of 5
            mp_temp = self.melt_pool_temp_save["temp"].values
            mp_depth = self.melt_pool_depth_save["depth"].values
            averaged_sequence_temp = []
            averaged_sequence_depth = []

        for i in range(0, len(mp_temp), 5):
            # Get the chunk of 5 elements
            chunk_temp = mp_temp[i:i+5]
            chunk_depth = mp_depth[i:i+5]
            # Calculate the average of the chunk and append to the averaged_sequence list
            
            # Try to take the average with the previous average term to smoothen the data
            averaging = True
            if averaging == True:
                average_chunk_temp = np.mean(chunk_temp)
                average_chunk_depth = np.mean(chunk_depth)
                
                if self.initial_average_temp == None:
                    self.initial_average_temp = np.mean(chunk_temp)
                    
                if self.initial_average_depth == None:
                    self.initial_average_depth = np.mean(chunk_depth)
                    
                averaged_sequence_temp.append(0.5*(self.initial_average_temp + average_chunk_temp))
                averaged_sequence_depth.append(0.5*(self.initial_average_depth + average_chunk_depth))
                self.initial_average_temp = average_chunk_temp
                self.initial_average_depth = average_chunk_depth
            else:
                average_chunk_temp = np.mean(chunk_temp)
                average_chunk_depth = np.mean(chunk_depth)
                averaged_sequence_temp.append(average_chunk_temp)
                averaged_sequence_depth.append(average_chunk_depth)

        # Convert the list to a numpy array or keep it as a list based on your preference
        return np.array(averaged_sequence_temp), np.array(averaged_sequence_depth)
            
    def run_sim_interval(self,laser_power):
        for i in range(self.sim_interval):
            self.GAMMA.run_onestep(laser_power)

            laser_on = 1
            self.melt_pool_temp_save, self.melt_pool_width_save, self.melt_pool_depth_save = get_melt_pool(laser_on, self.GAMMA, self.global_counter, self.melt_pool_temp_save, self.melt_pool_width_save, self.melt_pool_depth_save, self.solidus_temp)
            self.global_counter += 1
            
        averaging = True
        
        avg_mp_temp = np.mean(self.melt_pool_temp_save[-self.sim_interval:]["temp"].values)
        avg_mp_depth = np.mean(self.melt_pool_depth_save[-self.sim_interval:]["depth"].values)
        
        if averaging == True:
            avg_mp_temp_output = 0.5*(avg_mp_temp + self.initial_average_temp)
            avg_mp_depth_output = 0.5*(avg_mp_depth + self.initial_average_depth)
            self.initial_average_depth = avg_mp_depth
            self.initial_average_temp = avg_mp_temp
            if avg_mp_depth_output !=0 :
                self.non_zero_depth_buffer = avg_mp_depth_output
            else:
                avg_mp_depth_output = self.non_zero_depth_buffer
            
            return avg_mp_temp_output, avg_mp_depth_output   
        else:        
            return avg_mp_temp, avg_mp_depth


# Other functions

def compress_and_upload_with_progress(source_path, target_path):
    # Compress the file (assuming tar.gz compression)
    compressed_file_path = f"{source_path}.tar.gz"
    subprocess.run(['tar', '-czf', compressed_file_path, '-C', os.path.dirname(source_path), os.path.basename(source_path)], check=True)
    print(f"Compressed {source_path} to {compressed_file_path}")

    # Upload the compressed file
    subprocess.run(['rclone', 'copy', compressed_file_path, target_path], check=True)
    print(f"Uploaded {compressed_file_path} to {target_path}")
    
    # Delete the compressed file after uploading
    os.remove(compressed_file_path)
    print(f"Deleted local compressed file: {compressed_file_path}")

def delete_local_file(file_path):
    # Check if the path is a file or directory and delete accordingly
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
        print(f"Deleted directory: {file_path}")
    else:
        print(f"Path does not exist: {file_path}")


def get_melt_pool(laser_on,GAMMA, timestep, melt_pool_temp_save, melt_pool_width_save, melt_pool_depth_save, solidus_temp):
    if laser_on:
        melt_pool_width , melt_pool_temp, melt_pool_depth = get_meltpool_temp_width_depth(GAMMA, solidus_temp, timestep, None)
        melt_pool_temp_save = melt_pool_temp_save._append({'timestep':timestep,'temp':melt_pool_temp},ignore_index = True)
        melt_pool_width_save = melt_pool_width_save._append({'timestep':timestep,'width':melt_pool_width},ignore_index = True)
        melt_pool_depth_save = melt_pool_depth_save._append({'timestep':timestep,'depth':melt_pool_depth},ignore_index = True)

        return melt_pool_temp_save, melt_pool_width_save, melt_pool_depth_save
    else:
        melt_pool_temp_save = melt_pool_temp_save._append({'timestep':timestep,'temp':400},ignore_index = True)
        melt_pool_width_save = melt_pool_width_save._append({'timestep':timestep,'width':0},ignore_index = True)
        melt_pool_depth_save = melt_pool_depth_save._append({'timestep':timestep,'depth':0},ignore_index = True)
        return melt_pool_temp_save, melt_pool_width_save, melt_pool_depth_save
