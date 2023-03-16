import numpy as np
import pandas as pd
from utils.utils import convert_coordinates
import os

'''
Each function should take in folder containing autoIDd and ground truth data then 
read both and output df containing xyzrgb values, converted coordinates for unrolled 
visualization, and both ground truth and autoIDd labels (first and second rank)
'''

def proc_Chaud(folder):
    aut_file = None
    for file in os.listdir(folder):
        if file.startswith('autoID'):
            aut_file  = folder +'/'+file
        elif file.endswith('mark_w_names.csv'):
            gt_file = folder +'/' +file     

    gt = pd.read_csv(gt_file)
    gt = gt.rename(columns={'##x':'x'})

    df = pd.read_csv(aut_file)

    df= df.rename(columns={'aligned_x': 'X', 'aligned_y': 'Y', 'aligned_z': 'Z', 'aligned_R':'R', 'aligned_G':'G', 'aligned_B':'B'})
    df['ID'] = gt['ID']

    df_converted = convert_coordinates(df)

    return df_converted

def proc_FOCO(folder):
    aut_file = None
    gt_file = None
    for file in os.listdir(folder):
        if file.endswith("NP1.csv"):
            aut_file = folder+ '/' +file
        elif file.endswith("blobs.csv"):
            gt_file = folder+ '/' +file

    if (not aut_file) or (not gt_file):
        return None

    df = pd.read_csv(aut_file)
    gt = pd.read_csv(gt_file)

    df= df.rename(columns={'aligned_x': 'X', 'aligned_y': 'Y', 'aligned_z': 'Z', 'aligned_R':'R', 'aligned_G':'G', 'aligned_B':'B'})
    df['ID'] = gt['ID']

    df.loc[df['ID'].str[-1].isin(['?']),'ID'] = np.nan
    df_converted = convert_coordinates(df)
        
    return df_converted

def proc_NP(file):
    #TODO continue making this work (just copied and pasted right now)
    aut_file = 'data/NP_paper/all/autoID_output_'+file +'.csv' 
    gt_file = 'data/NP_paper/all/' +file+'.csv'

    df = pd.read_csv(aut_file)
    gt = pd.read_csv(gt_file, skiprows=7)
    
    df= df.rename(columns={'aligned_x': 'X', 'aligned_y': 'Y', 'aligned_z': 'Z', 'aligned_R':'R', 'aligned_G':'G', 'aligned_B':'B'})
    df['ID'] = gt['User ID']
    
    
    df_converted = convert_coordinates(df)

    return df_converted