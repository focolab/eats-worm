import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.optimize
import os
from utils.utils import convert_coordinates

class Atlas:

    def __init__(self, atlas_file = 'data/atlases/atlas_xx_rgb.mat', ganglia = 'data/atlases/neuron_ganglia.csv'):

        eng = matlab.engine.start_matlab()
        self.ganglia = pd.read_csv(ganglia)
        atlas_file = eng.load(atlas_file) 
        atlas = eng.getfield(atlas_file, 'atlas')
        head = eng.getfield(atlas, 'head') # only looking at head for now
        model = eng.getfield(head, 'model')
        mu = eng.getfield(model, 'mu') # Nx7 (XYZRGB)
        sigma = eng.getfield(model, 'sigma') # 7x7xN
        neurons = eng.getfield(head, 'N') # Nx1 strings of neuron names
        eng.quit()

        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)
        self.neurons = neurons
        self.neur_dict = {}
        self.df = None

    def create_dictionary(self):
        '''
        Create dictionary where keys are neuron IDs and values are dictionary of 
        neuron atlas attributes (xyz_mu, rgb_mu, xyz_sig, rgb_sig, class, ganglia)
        useful for 
        '''
        gang_dict = dict(zip(self.ganglia['neuron_class'].values, self.ganglia['ganglion'].values))
        
        for i, neuron in enumerate(self.neurons):

            ID = neuron

            if ID[-1] in ['L', 'R'] and ID[:-1]+'L' in self.neurons and ID[:-1]+'R' in self.neurons:        
                neurClass = ID[:-1]
                
            else:
                neurClass = ID

            gang = gang_dict.get(neurClass, 'Other')

            xyz_mu = self.mu[i,0:3]
            rgb_mu = self.mu[i,3:7]

            xyz_sigma = self.sigma[0:3, 0:3, i]
            rgb_sigma = self.sigma[3:7, 3:7, i]

            self.neur_dict[ID] = {'xyz_mu': xyz_mu, 'rgb_mu': rgb_mu, 'xyz_sigma':xyz_sigma, 'rgb_sigma': rgb_sigma, 'class':neurClass, 'ganglion':gang}

        return self.neur_dict

    def get_df(self):
        
        df_gangl = pd.DataFrame(self.ganglia)
        df_atlas = pd.DataFrame(self.mu, columns = ['X','Y','Z', 'R', 'G', 'B'])

        # find the LR paired neurons and assign neuron_class
        all_neurons = self.neurons
        neuron_class, is_LR, is_L, is_R = [], [], [], []
        for i in range(len(self.neurons)):
            ID = self.neurons[i]
            if ID[-1] in ['L', 'R'] and ID[:-1]+'L' in all_neurons and ID[:-1]+'R' in all_neurons:        
                neuron_class.append(ID[:-1])
                is_LR.append(1)
                if ID[-1] == 'L':
                    is_L.append(1)
                    is_R.append(0)
                if ID[-1] == 'R':
                    is_R.append(1)
                    is_L.append(0)
            else:
                neuron_class.append(ID)
                is_LR.append(0)
                is_L.append(0)
                is_R.append(0)

        df_atlas['neuron_class'] = neuron_class
        df_atlas['is_LR'] = is_LR
        df_atlas['is_L'] = is_L
        df_atlas['is_R'] = is_R
        df_atlas['ID'] = self.neurons

        # add ganglion column
        gang_dict = dict(zip(df_gangl['neuron_class'].values, df_gangl['ganglion'].values))
        df_atlas['ganglion'] = [gang_dict.get(k, 'other') for k in df_atlas['neuron_class']]  

        df_conv_atlas = convert_coordinates(df_atlas)

        self.df = df_conv_atlas

        return self.df