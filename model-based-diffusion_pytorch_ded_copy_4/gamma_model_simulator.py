# gamma_model_simulator.py

import os
import numpy as np
import cupy as cp
import gamma.interface as rs
from multiprocessing import Process
import time

class GammaModelSimulator:
    def __init__(self, input_data_dir, sim_dir_name, laser_file, VtkOutputStep=1., ZarrOutputStep=0.02, outputVtkFiles=True, verbose=True):
        self.input_data_dir = input_data_dir
        self.sim_dir_name = sim_dir_name
        self.laser_file = laser_file
        self.VtkOutputStep = VtkOutputStep
        self.ZarrOutputStep = ZarrOutputStep
        self.outputVtkFiles = outputVtkFiles
        self.verbose = verbose

        self.sim_itr = None

    def setup_simulation(self):
        self.sim_itr = rs.FeaModel(
                    input_data_dir=self.input_data_dir,
                    geom_dir=self.sim_dir_name,
                    laserpowerfile=self.laser_file,
                    VtkOutputStep=self.VtkOutputStep,
                    ZarrOutputStep=self.ZarrOutputStep,
                    outputVtkFiles=self.outputVtkFiles,
                    verbose=self.verbose)

    def run_simulation(self):
        if self.sim_itr:
            self.sim_itr.run()
        else:
            raise ValueError("Simulation is not setup yet. Call setup_simulation() first.")
