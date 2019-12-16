import numpy as np
import tifffile as tf
from pycpd import deformable_registration
from multipagetiff import *
from rainbow_tools import *
import time
from mpl_toolkits.mplot3d import Axes3D
from functools import partial

def visualize(iteration, error, X, Y, ax):
    #pdb.set_trace()
    plt.cla()
    #pdb.set_trace()
    temp = Y[:,3:6]
    temp[temp<=0] = 0.001
    Y[:,3:6] = temp
    #pdb.set_trace()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    #ax.scatter(X[:,0].reshape(-1),  X[:,1].reshape(-1), X[:,2].reshape(-1), c=X[:,3:6], label='Target')
    #ax.scatter(Y[:,0].reshape(-1),  Y[:,1].reshape(-1), Y[:,2].reshape(-1), c=Y[:,3:6], label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    #plt.savefig("deformable_xyzrgbw/" + str(iteration) + '.png', dpi = 300)
    plt.pause(0.1)
    #pdb.set_trace()
    #np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])


tf = MultiPageTiff('/Users/stevenban/Documents/Data/20190917/')
tf.set_numz(20)
tf.set_frames([0,1,2,3,4,5,6,7,8,9,10,11])
#import pdb
#pdb.set_trace()

tf.t = 0
im1 = tf.get_t()
tf.t = 20
im2 = tf.get_t()

#im2 = tf.get_t()


im1 = gaussian3d(im1)
im2 = gaussian3d(im2)


peaks1 = findpeaks3d(im1 * np.array(im1 > np.quantile(im1,.99)))
peaks2 = findpeaks3d(im2 * np.array(im2 > np.quantile(im2,.99)))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
callback = partial(visualize, ax=ax)

reg = deformable_registration(**{ 'X': peaks1, 'Y': peaks2 })
reg.register(callback)
plt.show()
#reg.register()

'''
fig = plt.figure()


ax = fig.add_subplot(111, projection='3d')
ax.scatter(peaks1[:,0],  peaks1[:,1], peaks1[:,2], color='red', label='Target')
ax.scatter(peaks2[:,0],  peaks2[:,1], peaks2[:,2], color='blue', label='Source')
plt.show()







def plotCenters(peaks1, peaks2):
    x = centers['x']
    y = centers['y']
    z = centers['z']
    r = centers['r']
    g = centers['g']
    b = centers['b']

    names = centers['name']

    colors = []
    for i in range(len(r)):
        colors.append('rgb('+str(r[i])+','+str(g[i])+','+str(b[i])+')')

    data = go.Scatter3d(
         x = x,
         y = y,
         z = z,
         mode ='markers',
         marker=dict(color=['red' for i in range(len(x))] + ['blue' for i in range(len(x))],
                   size=10, 
                   opacity = 0.8),
         hovertext = names,

    )
    data = [data]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            xaxis = dict(
            	title='X-axis (um)',
                showgrid=False,
            ),
            yaxis = dict(
            	title='Y-axis (um)',
                showgrid=False,
            ),
            zaxis = dict(
            	title='Z-axis (um)',
                showgrid=False,
            ),
            aspectmode='data'
        )
    )

    return go.Figure(data = data, layout = layout)

'''