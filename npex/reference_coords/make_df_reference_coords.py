#   hacking around
#   - combining atlas info (xyzrgb+ID) with neuron class and ganglion info
#   - conversion to cylindrical coordinates, to allow unrolled view
#
#   atlas_mu.csv scraped fron the Nejatbaksh CELL-ID repo
#   neuron_ganglia.csv retooled from the neuropal cell paper (SI spreadsheet)

import pdb
import pandas as pd
import napari
import numpy as np
import matplotlib.pyplot as plt

ganglia = 'neuron_ganglia.csv'
atlas = 'atlas_mu.csv'
IDCOL = 'names'
# basis vectors for new coord system (hand positioned by trial/errror)
v1 = np.asarray([[-40, 0, 0], [80, 0, -8]])
v2 = np.asarray([[-40, 0, 0], [-40.8, 0, -12]])
output_csv = 'df_reference_coords.csv'

#=======================================
# load atlas and ganglion info
df_atlas = pd.read_csv(atlas, index_col=0)
df_gangl = pd.read_csv(ganglia, index_col=0)

# find the LR paired neurons and assign neuron_class
all_neurons = df_atlas[IDCOL].values
neuron_class, is_LR, is_L, is_R = [], [], [], []
for i, row in df_atlas.iterrows():
    ID = row[IDCOL]
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

# add ganglion column
gang_dict = dict(zip(df_gangl['neuron_class'].values, df_gangl['ganglion'].values))
df_atlas['ganglion'] = [gang_dict.get(k, 'other') for k in df_atlas['neuron_class']]


# ### diagnostics (neuron_class should match between the two input dataframes)
# print(df_atlas.head())
# print(df_gangl.head())
# nc1 = list(set(df_atlas['neuron_class'].values))
# nc2 = list(set(df_gangl['neuron_class'].values))
# mis1 = sorted([x for x in nc1 if x not in nc2])
# mis2 = sorted([x for x in nc2 if x not in nc1])
# print()
# print('nc1 (%i):' % len(nc1))
# print(nc1)
# print()
# print('nc2 (%i):' % len(nc2))
# print(nc2)
# print()
# print('missing nc1 not in nc2:')
# print(mis1)
# print()
# print('missing nc2 not in nc1:')
# print(mis2)
# print()


# project into new cartesian coord system, then cyl coords
assert np.isclose(0, np.dot(v1[1]-v1[0], v2[1]-v2[0])) , 'v1 v2 should be orthogonal'
xyz = df_atlas[['X', 'Y', 'Z']].values
v1_norm = (v1[1]-v1[0])/np.linalg.norm(v1)
v2_norm = (v2[1]-v2[0])/np.linalg.norm(v2)
v3_norm = np.cross(v1_norm, v2_norm)
xnew = np.dot(xyz, v1_norm)
znew = np.dot(xyz-v1[0], v2_norm)
ynew = -np.dot(xyz-v1[0], v3_norm)
h = -xnew
r = np.sqrt(znew**2+ynew**2)
th = np.arctan2(ynew, -znew)/np.pi*180.0

# pack up and export
df_atlas['xcyl'] = xnew
df_atlas['ycyl'] = ynew
df_atlas['zcyl'] = znew
df_atlas['h'] = h
df_atlas['r'] = r
df_atlas['theta'] = th
df_atlas.to_csv(output_csv, float_format='%6g')




def plot_atlas_unrolled(df, png='plot-atlas-unrolled.png', show=True):
    """df needs: x/y/zcyl, ganglion, h, theta """
    ganglia = sorted(df['ganglion'].unique())
    gs_kw = dict(height_ratios=[1, 4])
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6, 12), gridspec_kw=gs_kw)
    for g in ganglia:
        dfg = df[df['ganglion'] == g]
        ax[0].plot(dfg['ycyl'], dfg['zcyl'], 'o', lw=0, mec='grey')
        ax[1].plot(dfg['theta'], dfg['h'], 'o', lw=0, mec='grey', label=g)
    ax[0].set_aspect('equal')
    ax[0].set_xlim([-3.5, 3.5])
    ax[0].set_ylim([-3, 3])
    ax[0].plot([0, 0], [0, 2.5], '--', color='grey')
    ax[0].plot(0, 0, 'x', color='k')
    ax[1].axvspan(-135, -45, edgecolor=None, color='lightgrey', alpha=0.4, zorder=0, lw=0)
    ax[1].axvspan(45, 135, edgecolor=None, color='lightgrey', alpha=0.4, zorder=0, lw=0)
    ax[1].axvline(-180, ls='--', color='grey')
    ax[1].axvline(180, ls='--', color='grey')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(png)
    if show:
        plt.show()

def plot_atlas_2d_views(df_atlas, png='plot-atlas.png', show=True):
    fig = plt.figure(figsize=(15,6))
    ax = [plt.subplot(211)]
    ax.append(plt.subplot(212, sharex=ax[0]))
    #ax[0].plot(df_atlas['X'], df_atlas['Z'], 'o', mec='grey', ms=15)
    for i, row in df_atlas.iterrows():
        ax[0].plot(row['X'], -row['Z'], 'o', color='lightblue', mec='grey', mew=1, ms=10)
        if i % 5 == 0:
            ax[0].text(row['X'], -row['Z'], '%i'%i, ha='center', va='center', fontsize=8, color='k')
    ax[0].set_aspect('equal')
    ax[0].grid()
    ax[0].set_ylabel('Z')
    #ax[1].plot(df_atlas['X'], df_atlas['Y'], 'o', mec='grey')
    for i, row in df_atlas.iterrows():
        ax[1].plot(row['X'], row['Y'], 'o', color='lightblue', mec='grey', mew=1, ms=10)
        if i % 5 == 0:
            ax[1].text(row['X'], row['Y'], '%i'%i, ha='center', va='center', fontsize=8, color='k')
    ax[1].set_aspect('equal')
    ax[1].grid()
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    plt.tight_layout()
    plt.savefig(png)
    if show:
        plt.show()



plot_atlas_unrolled(df_atlas, png='plot-atlas-unrolled.png', show=False)
plot_atlas_2d_views(df_atlas, png='plot-atlas-2d-views.png', show=False)


# v = napari.Viewer()
# p = df[['X', 'Y', 'Z']].values
# p1 = v.add_points(p, size=1) #, **dd)
# p2 = v.add_points(v1, size=2.5) #, **dd)
# p3 = v.add_points(v2, size=2.5) #, **dd)
# pdb.set_trace()
