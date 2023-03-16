import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from utils.utils import covar_to_coord
import scipy.optimize
import os
import process.file as fil
import utils.utils as uti

#use patches.ellipse to get ellipses

def plot_atlas_RGB(df, sigma):

    rgb_mu = np.asarray(df[['R', 'G', 'B']])
    rgb_sigma = sigma[3:7,3:7, :]

    fig, axs = plt.subplots(1 , 3, figsize=(24,48))

    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(-10, 30)
        ax.set_ylim(-10, 30)

    for n in range(rgb_sigma.shape[2]):
        
        rgl1, rgl2, rgtheta = covar_to_coord(rgb_sigma[[0,1],:,n][:,[0,1]])
        rbl1, rbl2, rbtheta = covar_to_coord(rgb_sigma[[0,2],:,n][:,[0,2]])
        gbl1, gbl2, gbtheta = covar_to_coord(rgb_sigma[[1,2],:,n][:,[1,2]])

        rmu = rgb_mu[n, 0]
        gmu = rgb_mu[n, 1]
        bmu = rgb_mu[n, 2]
        
        #looking at only half a std to make it easier to visualize 

        rg_ellipse = Ellipse((rmu,gmu), width =rgl1*2, height = rgl2*2, angle=rgtheta*180/np.pi, alpha=0.05, edgecolor='orange', facecolor='orange')
        axs[0].add_patch(rg_ellipse)
        rb_ellipse = Ellipse((rmu, bmu), width =rbl1*2, height = rbl2*2, angle=rbtheta*180/np.pi, alpha=0.05, edgecolor='magenta', facecolor='magenta')
        axs[1].add_patch(rb_ellipse)
        gb_ellipse = Ellipse((gmu, bmu), width =gbl1*2, height = gbl2*2, angle=gbtheta*180/np.pi, alpha=0.05, edgecolor='cyan', facecolor='cyan')
        axs[2].add_patch(gb_ellipse)

    axs[0].set_title('red-green')
    axs[0].set_xlabel('red')
    axs[0].set_ylabel('green')
    axs[1].set_title('red-blue')
    axs[1].set_xlabel('red')
    axs[1].set_ylabel('blue')
    axs[2].set_title('green-blue')
    axs[2].set_xlabel('green')
    axs[2].set_ylabel('blue')


    plt.show()

def plot_RGB_super(df, sigma, df_data):

    rgb_data = np.asarray(df_data[['R', 'G', 'B']])

    rgb_mu = np.asarray(df[['R', 'G', 'B']])
    rgb_sigma = sigma[3:7,3:7, :]

    fig, axs = plt.subplots(1 , 3, figsize=(24,48))

    for ax in axs:
        ax.set_aspect('equal')

    for n in range(rgb_sigma.shape[2]):
        
        rgl1, rgl2, rgtheta = covar_to_coord(rgb_sigma[[0,1],:,n][:,[0,1]])
        rbl1, rbl2, rbtheta = covar_to_coord(rgb_sigma[[0,2],:,n][:,[0,2]])
        gbl1, gbl2, gbtheta = covar_to_coord(rgb_sigma[[1,2],:,n][:,[1,2]])

        rmu = rgb_mu[n, 0]
        gmu = rgb_mu[n, 1]
        bmu = rgb_mu[n, 2]

        rg_ellipse = Ellipse((rmu,gmu), width =rgl1*2, height = rgl2*2, angle=rgtheta*180/np.pi, alpha=0.05, edgecolor='orange', facecolor='orange')
        axs[0].add_patch(rg_ellipse)
        rb_ellipse = Ellipse((rmu, bmu), width =rbl1*2, height = rbl2*2, angle=rbtheta*180/np.pi, alpha=0.05, edgecolor='magenta', facecolor='magenta')
        axs[1].add_patch(rb_ellipse)
        gb_ellipse = Ellipse((gmu, bmu), width =gbl1*2, height = gbl2*2, angle=gbtheta*180/np.pi, alpha=0.05, edgecolor='cyan', facecolor='cyan')
        axs[2].add_patch(gb_ellipse)
     
    colors_min = np.amin(rgb_data, axis=0)
    colors_max = np.amax(rgb_data, axis=0)
    color_norm = np.divide(rgb_data-colors_min, colors_max-colors_min)
    
    axs[0].scatter(rgb_data[:,0], rgb_data[:,1], c=color_norm)
    axs[1].scatter(rgb_data[:,0], rgb_data[:,2], c=color_norm)
    axs[2].scatter(rgb_data[:,1], rgb_data[:,2], c=color_norm)

    fontsize= 16
    axs[0].set_title('red-green', fontsize=fontsize)
    axs[0].set_xlabel('red', fontsize=fontsize)
    axs[0].set_ylabel('green', fontsize=fontsize)
    axs[1].set_title('red-blue', fontsize=fontsize)
    axs[1].set_xlabel('red', fontsize=fontsize)
    axs[1].set_ylabel('blue', fontsize=fontsize)
    axs[2].set_title('green-blue', fontsize=fontsize)
    axs[2].set_xlabel('green', fontsize=fontsize)
    axs[2].set_ylabel('blue', fontsize=fontsize)


    plt.show()

def plot_atlas_2d_views(df_atlas, sigma, df_data):

    xyz_data = df_data[['X', 'Y', 'Z']]
    rgb_data = np.asarray(df_data[['R', 'G', 'B']])

    xyz_sigma = sigma[0:3, 0:3,:]

    fig = plt.figure(figsize=(15,6))
    ax = [plt.subplot(211)]
    ax.append(plt.subplot(212, sharex=ax[0]))

    #ax[0].plot(df_atlas['X'], df_atlas['Z'], 'o', mec='grey', ms=15)
    for i, row in df_atlas.iterrows():
        xzl1, xzl2, xztheta = covar_to_coord(xyz_sigma[[0,2],:,i][:,[0,2]]) 
        xyl1, xyl2, xytheta = covar_to_coord(xyz_sigma[[0,1],:,i][:,[0,1]])

        xz_ellipse = Ellipse((row['X'], row['Z']), width = xzl1*2, height=xzl2*2, angle=xztheta*180/np.pi, alpha=0.2, edgecolor = 'blue', facecolor='blue', linestyle='-')
        xy_ellipse = Ellipse((row['X'], row['Y']), width = xyl1*2, height=xyl2*2, angle=xytheta*180/np.pi, alpha=0.2, edgecolor = 'blue', facecolor='blue', linestyle='-')
        ax[0].add_patch(xz_ellipse)
        ax[1].add_patch(xy_ellipse)
        
    colors_min = np.amin(rgb_data, axis=0)
    colors_max = np.amax(rgb_data, axis=0)
    color_norm = np.divide(rgb_data-colors_min, colors_max-colors_min)
  
        
    ax[0].scatter(xyz_data['X'], xyz_data['Z'], c=color_norm)
    ax[1].scatter(xyz_data['X'], xyz_data['Y'], c=color_norm)
  
    ax[0].set_aspect('equal')
    ax[0].grid()
    ax[0].set_ylabel('Z')
    ax[0].autoscale_view()

    ax[1].set_aspect('equal')
    ax[1].grid()
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].autoscale_view()

    #plt.tight_layout()
    plt.show()

def plot_atlas_unrolled(df, png='plot-atlas-unrolled.png', show=True):
    """df needs: x/y/zcyl, ganglion, h, theta """

    ganglia = sorted(df['ganglion'].unique())
    gs_kw = dict(height_ratios=[1, 4])
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6, 12), gridspec_kw=gs_kw)

    for g in ganglia:
        dfg = df[df['ganglion'] == g]
        ax[0].plot(dfg['ycyl'], dfg['zcyl'], 'o', lw=0,markerfacecolor='None')
        ax[1].plot(dfg['theta'], dfg['h'], 'o', lw=0, label=g, markerfacecolor='None')

    ax[0].set_aspect('equal')
    ax[0].set_xlim([-4, 4])
    ax[0].set_ylim([-4, 4])
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

def plot_atlas_unrolled_superimpose(df_atlas, df_data, png='plot-atlas-unrolled.png', show=True):
    """df needs: x/y/zcyl, ganglion, h, theta """

    rgb_data = np.asarray(df_data[['R', 'G', 'B']])

    ganglia = sorted(df_atlas['ganglion'].unique())
    gs_kw = dict(width_ratios=[1, 2])
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8), gridspec_kw=gs_kw)

    for g in ganglia:
        dfg = df_atlas[df_atlas['ganglion'] == g]
        ax[0].plot(dfg['ycyl'], dfg['zcyl'], 'o', lw=0,markerfacecolor='None')
        ax[1].plot(dfg['theta'], dfg['h'], 'o', lw=0, label=g, markerfacecolor='None')
        
    colors_min = np.amin(rgb_data, axis=0)
    colors_max = np.amax(rgb_data, axis=0)
    color_norm = np.divide(rgb_data-colors_min, colors_max-colors_min)

    ax[0].set_aspect('equal')
    ax[0].set_xlim([-4, 4])
    ax[0].set_ylim([-4, 4])
    ax[0].plot([0, 0], [0, 2.5], '--', color='grey')
    ax[0].plot(0, 0, 'x', color='k')
    ax[0].scatter(df_data['ycyl'], df_data['zcyl'], c=color_norm)

    ax[1].axvspan(-135, -45, edgecolor=None, color='lightgrey', alpha=0.4, zorder=0, lw=0)
    ax[1].axvspan(45, 135, edgecolor=None, color='lightgrey', alpha=0.4, zorder=0, lw=0)
    ax[1].axvline(-180, ls='--', color='grey')
    ax[1].axvline(180, ls='--', color='grey')
    ax[1].scatter(df_data['theta'], df_data['h'], c=color_norm)
    ax[1].legend()
    ax[1].set_xlabel('Theta')
    ax[1].set_ylabel('X')

    plt.tight_layout()
    plt.savefig(png)
    if show:
        plt.show()

def plot_all(df_atlas, sigma, df_data):

    plot_RGB_super(df_atlas, sigma, df_data)
    plot_atlas_unrolled_superimpose(df_atlas, df_data)
    plot_atlas_2d_views(df_atlas, sigma, df_data)

def plot_atlas_colors(df_atlas):

    rgb_mu = np.asarray(df_atlas[['R', 'G', 'B']])
    
    ganglia = sorted(df_atlas['ganglion'].unique())
    gs_kw = dict(width_ratios=[1, 2])
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8), gridspec_kw=gs_kw)

    atlas_color_min = np.amin(rgb_mu, axis=0)
    atlas_color_max = np.amax(rgb_mu, axis=0)
    atlas_color_norm = (rgb_mu-atlas_color_min)/(atlas_color_max-atlas_color_min)
    
    
    ax[0].scatter(df_atlas['ycyl'], df_atlas['zcyl'], edgecolor=atlas_color_norm, facecolor='none')
    ax[1].scatter(df_atlas['theta'], df_atlas['h'], edgecolor=atlas_color_norm, facecolor='none')
    
    plt.tight_layout()
    plt.show()

def plot_unrolled_acc(df_atlas, df_data):

    rgb_mu = np.asarray(df_atlas[['R', 'G', 'B']])
    rgb_data = np.asarray(df_data[['R', 'G', 'B']])
    
    ganglia = sorted(df_atlas['ganglion'].unique())
    gs_kw = dict(width_ratios=[1, 2])
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8), gridspec_kw=gs_kw)

    atlas_color_min = np.amin(rgb_mu, axis=0)
    atlas_color_max = np.amax(rgb_mu, axis=0)
    atlas_color_norm = (rgb_mu-atlas_color_min)/(atlas_color_max-atlas_color_min)
    
    ax[0].scatter(df_atlas['ycyl'], df_atlas['zcyl'], edgecolor=atlas_color_norm, facecolor='none')
    ax[1].scatter(df_atlas['theta'], df_atlas['h'], edgecolor=atlas_color_norm, facecolor='none')
    
    #for g in ganglia:
    #    dfg = df_atlas[df_atlas['ganglion'] == g]
    #    ax[0].plot(dfg['ycyl'], dfg['zcyl'], 'o', lw=0,markerfacecolor='None')
    #    ax[1].plot(dfg['theta'], dfg['h'], 'o', lw=0, label=g, markerfacecolor='None')
        
    colors_min = np.amin(rgb_data, axis=0)
    colors_max = np.amax(rgb_data, axis=0)
    color_norm = (rgb_data-colors_min)/(colors_max-colors_min)


    ax[0].set_aspect('equal')
    ax[0].set_xlim([-4, 4])
    ax[0].set_ylim([-4, 4])
    ax[0].plot([0, 0], [0, 2.5], '--', color='grey')
    ax[0].plot(0, 0, 'x', color='k')
    ax[0].scatter(df_data['ycyl'], df_data['zcyl'], c=color_norm)

    ax[1].axvspan(-135, -45, edgecolor=None, color='lightgrey', alpha=0.4, zorder=0, lw=0)
    ax[1].axvspan(45, 135, edgecolor=None, color='lightgrey', alpha=0.4, zorder=0, lw=0)
    ax[1].axvline(-180, ls='--', color='grey')
    ax[1].axvline(180, ls='--', color='grey')
    ax[1].scatter(df_data['theta'], df_data['h'], c=color_norm)
    
    IDs = np.asarray(df_data['ID'])
    
    for i, txt in enumerate(IDs):
        #ax[1].annotate(IDs[i], (df_data.loc[df_data['ID']==IDs[i]]['theta'], df_data.loc[df_data['ID']==IDs[i]]['h']))
        #ax[1].annotate(IDs[i], (df_atlas.loc[df_atlas['ID']==IDs[i]]['theta'],df_atlas.loc[df_atlas['ID']==IDs[i]]['h']))
        p1 = [df_data.loc[df_data['ID']==IDs[i]]['theta'], df_data.loc[df_data['ID']==IDs[i]]['h']]
        p2 = [df_atlas.loc[df_atlas['ID']==IDs[i]]['theta'],df_atlas.loc[df_atlas['ID']==IDs[i]]['h']]
        x, y = [p1[0], p2[0]], [p1[1], p2[1]]
        plt.plot(x,y)

    plt.tight_layout()
    plt.show()

def check_alignment_FOCO(data, df_atlas, sigma):
    '''
    Currently only setup for FOCO data folder
    TODO to add functionality for other types of data as well
    '''
       
    alignvals = []
    accvals = []
    names = []
    corr_dict = {}
    ID_dict = {}

    folders = os.listdir('data/'+data)

    for f in folders:
        
        if not os.path.isdir('data/'+data + '/'+f):
            continue 
        # want to plot accuracy vs alignment cost (both xyz only and xyzrgb)
        
        df_data = fil.proc_FOCO('data/'+data + '/'+f)

        cost_xyz, cost_rgb = uti.calc_costs(df_atlas, sigma, df_data)

        IDd, correctID, correctfirstsecond, correct_df, correcttop2 = uti.check_accuracy(df_data)

        corrIDs = np.asarray(correct_df['ID'])
        for txt in corrIDs:
            if txt not in corr_dict.keys():
                corr_dict[txt] = 1
            else:
                corr_dict[txt] += 1

        IDs = np.asarray(df_data['ID'])
        for txt in IDs:
            if txt not in ID_dict.keys():
                ID_dict[txt] = 1
            else:
                ID_dict[txt] +=1


        alignvals.append([cost_xyz, cost_rgb, correctID])
        accvals.append([IDd, correctID, correctfirstsecond])
        names.append(f)
    
    for key in corr_dict.keys():
        corr_dict[key] = corr_dict[key]/ID_dict[key]

    alignvals = np.asarray(alignvals)
    accvals = np.asarray(accvals)

    print(dict(sorted(corr_dict.items(), key=lambda item: item[1], reverse=True)))
    #print(dict(sorted(ID_dict.items(), key=lambda item: item[1], reverse=True)))

    fig, ax = plt.subplots(ncols=1, nrows=3)

    ax[0].scatter(alignvals[:,0], alignvals[:,2], label = 'xyz alignment cost')
    ax[0].scatter(alignvals[:,1], alignvals[:,2], label = 'rgb alignment cost')
    ax[0].legend()
    ax[0].set_xlabel('Alignment cost - mahalanobis distance')
    ax[0].set_ylabel('Assignment accuracy')
    ax[1].scatter(alignvals[:,0]/alignvals[:,1], alignvals[:,2])
    ax[1].set_xlabel('Ratio of rgb alignment to xyz alignment')
    ax[1].set_ylabel('Assignment accuracy')
    ax[2].scatter(accvals[:,0], accvals[:,2])
    ax[2].set_xlabel('Percent of neurons IDd by user')
    ax[2].set_ylabel('Assignment accuracy (first and second guess)')

    for i, txt in enumerate(names):
        ax[0].annotate(txt, (alignvals[i,0], alignvals[i,2]), size=6)
        ax[1].annotate(txt, (alignvals[i,0]/alignvals[i,1], alignvals[i,2]), size=6)
        ax[2].annotate(txt, (accvals[i,0], accvals[i,2]), size=6)
    fig.tight_layout()
    plt.show()
    
def plot_RGB_super_acc(df, sigma, df_data, indices):

    rgb_mu = np.asarray(df[['R', 'G', 'B']])
    rgb_sigma = sigma[3:7,3:7, :]

    fig, axs = plt.subplots(1 , 3, figsize=(24,48))

    for ax in axs:
        ax.set_aspect('equal')

    for n in range(rgb_sigma.shape[2]):
        
        rgl1, rgl2, rgtheta = covar_to_coord(rgb_sigma[[0,1],:,n][:,[0,1]])
        rbl1, rbl2, rbtheta = covar_to_coord(rgb_sigma[[0,2],:,n][:,[0,2]])
        gbl1, gbl2, gbtheta = covar_to_coord(rgb_sigma[[1,2],:,n][:,[1,2]])

        rmu = rgb_mu[n, 0]
        gmu = rgb_mu[n, 1]
        bmu = rgb_mu[n, 2]

        rg_ellipse = Ellipse((rmu,gmu), width =rgl1*2, height = rgl2*2, angle=rgtheta*180/np.pi, alpha=0.05, edgecolor='orange', facecolor='orange')
        axs[0].add_patch(rg_ellipse)
        rb_ellipse = Ellipse((rmu, bmu), width =rbl1*2, height = rbl2*2, angle=rbtheta*180/np.pi, alpha=0.05, edgecolor='magenta', facecolor='magenta')
        axs[1].add_patch(rb_ellipse)
        gb_ellipse = Ellipse((gmu, bmu), width =gbl1*2, height = gbl2*2, angle=gbtheta*180/np.pi, alpha=0.05, edgecolor='cyan', facecolor='cyan')
        axs[2].add_patch(gb_ellipse)
    
    rgb_data = np.asarray(df_data[['R', 'G', 'B']])

    rgb_data = rgb_data[indices]

    colors_min = np.amin(rgb_data, axis=0)
    colors_max = np.amax(rgb_data, axis=0)
    color_norm = np.divide(rgb_data-colors_min, colors_max-colors_min)

    axs[0].scatter(rgb_data[:,0], rgb_data[:,1], c=color_norm)
    axs[1].scatter(rgb_data[:,0], rgb_data[:,2], c=color_norm)
    axs[2].scatter(rgb_data[:,1], rgb_data[:,2], c=color_norm)

    fontsize= 16
    axs[0].set_title('red-green', fontsize=fontsize)
    axs[0].set_xlabel('red', fontsize=fontsize)
    axs[0].set_ylabel('green', fontsize=fontsize)
    axs[1].set_title('red-blue', fontsize=fontsize)
    axs[1].set_xlabel('red', fontsize=fontsize)
    axs[1].set_ylabel('blue', fontsize=fontsize)
    axs[2].set_title('green-blue', fontsize=fontsize)
    axs[2].set_xlabel('green', fontsize=fontsize)
    axs[2].set_ylabel('blue', fontsize=fontsize)


    plt.show()