import numpy as np
#from pycpd import deformable_registration 
import pdb
import time
import scipy.spatial
from scipy.optimize import linear_sum_assignment
import copy

def reg_peaks(im, peaks, thresh = 36, anisotropy = (6,1,1)):
    """
    function that drops peaks that are found too close to each other (Not used by Spool or Thread)
    """
    peaks = np.array(peaks)
    anisotropy = np.array(anisotropy,dtype=int)
    for i in range(len(anisotropy)):
        peaks[:,i]*=anisotropy[i]

    diff = scipy.spatial.distance.cdist(peaks,peaks,metric='euclidean')
    complete = False

    while not complete:
        try:
            x,y = np.where(diff == np.min(diff[diff!=0]))
            if diff[x,y] >= thresh:
                complete = True
            else:
                peak1 = peaks[x]
                peak2 = peaks[y]

                if im[peak1] > im[peak2]:
                    diff = np.delete(diff, y, axis = 0)
                    diff = np.delete(diff, y, axis = 1)
                    peaks = np.delete(peaks, y, axis = 0)
                else:
                    diff = np.delete(diff, x, axis = 0)
                    diff = np.delete(diff, x, axis = 1)
                    peaks = np.delete(peaks, x, axis = 0)
        except:
            complete = True

    for i in range(len(anisotropy)):
        peaks[:,i]= peaks[:,i]/anisotropy[i]
    return peaks.astype(int)

class Spool:
    """
    New class for spool, for 'flocking' behavior
    """

    def __init__(self, *args, **kwargs):
        self.threads = []

        self.blob_dist_thresh = 7

        if kwargs.get('blob_dist_thresh'):
            self.blob_dist_thresh = kwargs.get('blob_dist_thresh')
        elif len(args) != 0:
            self.blob_dist_thresh = args[0]

        if kwargs.get('max_t'):
            self.maxt = kwargs.get('max_t')
        elif len(args) == 2:
            self.maxt = args[1]


        self.predict = kwargs.get('predict')


        self.t = None
        self.dvec = np.zeros((self.maxt,3))
 
    def reel(self, positions, anisotropy = (6,1,1), t=0, offset=np.array([0,0,0])):

        # if no threads already exist, add all incoming points to new threads them and positions tracker
        if self.threads == []:
            for i in range(len(positions)):
                self.threads.append(Thread(positions[i], maxt = self.maxt))
            self.update_positions()
            self.predictions = copy.copy(self.positions)
            self.t = 1
        # if threads already exist
        else:
            # match points based on a max-threshold euclidean distance based matching
            #try:
            self.predictions = self.predictions + offset
            for i in range(len(anisotropy)):
                self.predictions[:,i]=self.predictions[:,i]*anisotropy[i]
                positions[:,i]=positions[:,i]*anisotropy[i]
            diff = scipy.spatial.distance.cdist(self.predictions, positions, metric='euclidean')
            for i in range(len(anisotropy)):        
                self.predictions[:,i]=self.predictions[:,i]/anisotropy[i]
                positions[:,i]=positions[:,i]/anisotropy[i]
            
            matchings, unmatched, newpoints = self.calc_match(diff, self.blob_dist_thresh)

            #dvec[self.t] = np.zeros(self.threads[0].get_position_mostrecent().shape)
            for match in matchings:
                self.dvec[self.t] -= self.threads[match[0]].get_position_mostrecent()-positions[match[1]]
                self.threads[match[0]].update_position(positions[match[1]], t = self.t)
            if matchings.any():
                self.dvec[self.t] *= 1/len(matchings)
            else:
                self.dvec[self.t] = 0
            #print(self.dvec[self.t])
            for match in unmatched:
                self.threads[match].update_position(self.threads[match].get_position_mostrecent() + self.dvec[self.t], t=self.t)

            for point in newpoints:
                self.threads.append(Thread(positions[point], t=self.t, maxt = self.maxt))
            
            self.update_positions()
            self.update_predictions()
            self.t += 1
    def calc_match(self, mat, thresh):
        """
        Calculate matches based on distance matrix
        input
        - mat:         incoming distance matrix
        - thresh:     max-dist threshold
        """
        '''
        unmatched = []
        newpoints = []
        orig_nd_r = np.array(range(mat.shape[0]))
        orig_nd_c = np.array(range(mat.shape[1]))
        for i in range(mat.shape[0]):
            if np.min(mat[i,:]) >= thresh:
                #mat = np.delete(mat, i, 0)
                unmatched.append(i)
        for j in range(mat.shape[1]):
            if np.min(mat[:,j]) >= thresh:
                #mat = np.delete(mat, i, 1)
                newpoints.append(j)
        unmatched.sort(reverse = True)
        newpoints.sort(reverse = True)
        for i in unmatched:
            mat = np.delete(mat,i,0)
            orig_nd_r = np.delete(orig_nd_r, i, 0)
        for i in newpoints:
            mat = np.delete(mat,i,1)
            orig_nd_c = np.delete(orig_nd_c, i, 0)
        #print(mat.shape)
        row,column = linear_sum_assignment(mat)
        row, column = orig_nd_r[row], orig_nd_c[column]
        matchings = np.array([row, column]).T
        #print(matchings)
        '''
        matchings = []
        if np.size(mat):
            for i in range(mat.shape[0]): #iterate over existing points
                if np.min(mat[i,:]) < thresh:
                    index = np.where(mat[i,:] == np.min(mat[i,:]))[0][0]
                    matchings.append([i,index])

                    mat[i,:] = 10000
                    mat[:,index] = 10000
                else: pass
            
        matchings = np.array(matchings)
        if matchings.any():
            unmatched = list(set(range(mat.shape[0]))-set(matchings[:,0]))
            newpoints = list(set(range(mat.shape[1]))-set(matchings[:,1]))
        else:
            unmatched = list(range(mat.shape[0]))
            newpoints = list(range(mat.shape[1]))

        return matchings, unmatched, newpoints

    def update_positions(self):
        """
        updates positions based on threads of matched points
        """
        self.positions = np.zeros((len(self.threads), self.threads[0].get_position_mostrecent().shape[0]))

        for i in range(len(self.threads)):
            self.positions[i] = self.threads[i].get_position_mostrecent()


    def update_predictions(self):
        self.predictions = np.zeros((len(self.threads), self.threads[0].get_position_mostrecent().shape[0]))


        if self.predict:
            for i in range(len(self.threads)):
                self.predictions[i] = self.threads[i].get_position_mostrecent() + self.dvec[self.t]
                
                if len(self.threads[i].t) > 1:
                    self.predictions[i] = self.threads[i].get_position_t(self.threads[i].t[-1])+ self.dvec[self.t]
                else:
                    self.predictions[i] = self.threads[i].get_position_mostrecent() + self.dvec[self.t]
        else:
            for i in range(len(self.threads)):
                self.predictions[i] = self.threads[i].get_position_mostrecent()
    def infill(self):
        for i in range(len(self.threads)):
            if self.threads[i].t[0]==0:
                pass
            else:
                inferred = self.threads[i].get_position_t(self.threads[i].t[0])
                for j in reversed(range(self.threads[i].t[0])):
                    inferred = inferred - self.dvec[j]
                    self.threads[i].infill(inferred)


class Thread:
    """
    Class for single blob thread. Contains the following 
    Properties:
        - positions:     list of positions that the current blob was found at, indexed the same as image indexing
        - t:             list of time points that the current blob was found at, i.e. position[i] was found at time point t[i]

    Methods:
        - get_position_mostrecent():     returns most recent position
        - update_position(position, t):    updates list of positions and time point; default t is most recent t + 1
        - get_positions_t(t):             returns position at time point specified, and if blob wasn't found at that time, then returns the position at the most recent time before time point specified

    Most recent edit: 
    10/23/2019
    """
    def __init__(self, position = [], t = 0, **kwargs):
        

        maxt = kwargs.get('maxt')
        self.positions = np.zeros((maxt,3))

        #self.positions = []
        self.t = []
        if position != []:
            self.positions[t] = np.array(position)
            #self.t = t + 1
            #self.positions.append(np.array(position))
            self.t.append(int(t))

    def get_position_mostrecent(self):
        """    
        returns most recent position
        """
        return self.positions[self.t[-1]]

    def update_position(self, position, t = 0):
        """
        takes in position and updates the thread with the position
        """

        if self.t != []: #if self.positions exist

            if len(position) == len(self.positions[0]): #append only if positions is same dimensions
                self.positions[t] = np.array(position)
                self.t.append(t)
                #self.positions.append(np.array(position))
            else:
                return False

        else:
            self.positions[t] = np.array(position)
            self.t.append(t)
            #self.positions.append(np.array(position))

        '''
        # if passed time argument
        if t:
            if not self.t:# if self.t doesn't yet exist
                self.t.append(t)
            elif self.t[-1] < t: #only adding timepoints in increasing order
                self.t.append(t)
            else: #f not increasing
                return False
        
        # if not passed time argument, and no previous time points exist
        elif not t and not self.t:
            self.t.append(0)

        # if not passed time argument, but previous points exist
        else:
            self.t.append(self.t[-1]+1)
        '''
    def infill(self, position):
        self.t.insert(0,self.t[0]-1)
        self.positions[self.t[0]] = np.array(position)
        #self.positions.insert(0,position)

    def get_position_t(self, t = 0):
        """
        get position at time point.
        - if time point exists, return position
        - if time point doesnt exist:
            - if time point is before thread was initialized, return False
            - if time point larger than largest time point, take the last time point
            - if time point not in thread but smaller than largest time point, update with the last-observed position before the time specified
        """
        t = int(t)
        if not self.t:
            return False
        
        elif t in self.t: #if time point exists, then just return that time point
            return self.positions[t]
        
        elif t < self.t[0]: #if time point doesn't exist, and time point less than smallest point this blob thread was initialized
            return False

        elif t > self.t[-1]: #if time point larger than largest time point, just take the last time point
            return self.positions[-1]

        else: #else find closest timepoint that came before time specified
            for i in range(len(self.t)-1):
                if self.t[i] <= t and self.t[i+1]>=t:
                    return self.positions[i]
            return self.positions[-1]




'''
DEPRECATED CODE:
    - CPD transform in between time points
    - Kalman filtering of neuron positions


def reg_transform(Y, TY, thresh=6, anisotropy = (6,1,1)):
    Y, TY = np.array(Y), np.array(TY)
    anisotropy = np.array(anisotropy,dtype=int)
    for i in range(len(anisotropy)):
        Y[:,i]*=anisotropy[i]
        TY[:,i]+=anisotropy[i]
    
    for i in range(len(TY)):
        diff = np.linalg.norm(TY[i]-Y[i])
        if diff < thresh:
            pass
        else:
            TY[i] = Y[i] + thresh*(TY[i]-Y[i])/diff
    for i in range(len(anisotropy)):
        TY[:,i]= TY[:,i]/anisotropy[i]
    return TY
    
def reg_peaks(im, peaks, thresh = 25, anisotropy = (6,1,1)):
    """
    function that drops peaks that are found too close to each other 
    """
    peaks = np.array(peaks)
    anisotropy = np.array(anisotropy,dtype=int)
    for i in range(len(anisotropy)):
        peaks[:,i]*=anisotropy[i]

    diff = scipy.spatial.distance.cdist(peaks,peaks,metric='euclidean')
    complete = False

    while not complete:
        try:
            x,y = np.where(diff == np.min(diff[diff!=0]))
            if diff[x,y] >= thresh:
                complete = True
            else:
                peak1 = peaks[x]
                peak2 = peaks[y]

                if im[peak1] > im[peak2]:
                    diff = np.delete(diff, y, axis = 0)
                    diff = np.delete(diff, y, axis = 1)
                    peaks = np.delete(peaks, y, axis = 0)
                else:
                    diff = np.delete(diff, x, axis = 0)
                    diff = np.delete(diff, x, axis = 1)
                    peaks = np.delete(peaks, x, axis = 0)
        except:
            complete = True

    for i in range(len(anisotropy)):
        peaks[:,i]= peaks[:,i]/anisotropy[i]
    return peaks.astype(int)



def calc_match(mat, thresh):
    matchings = []
    for i in range(mat.shape[0]): #iterate over existing points
        if np.min(mat[i,:]) < thresh:
            index = np.where(mat[i,:] == np.min(mat[i,:]))[0][0]
            matchings.append([i,index])

            mat[i,:] = 10000
            mat[:,index] = 10000
        else: pass

    matchings = np.array(matchings)
    unmatched = list(set(range(mat.shape[0]))-set(matchings[:,0]))
    newpoints = list(set(range(mat.shape[1]))-set(matchings[:,1]))
    return matchings, unmatched, newpoints


class Spool:
    """
    Class for holding/managing multiple blob threads
    """
    def __init__(self):
        self.threads = []
        self.t = None
        self.blob_dist_thresh = 10
    def reel(self, positions, t=0):

        # if no threads already exist, add all of them and update positions
        if self.threads == []:
            for i in range(len(positions)):
                self.threads.append(Thread(positions[i]))
            self.update_positions()
            self.t = 0
        # if threads already exist
        else:
            ## register current positions to incoming positions
            a = time.time()
            _reg = deformable_registration(**{ 'X': self.positions, 'Y': positions })
            _reg.register()
            Y = _reg.TY
            del _reg
            #print('Registration: ',time.time() - a)
            #pdb.set_trace()
            ## regularize/limit the transformation
            Y = reg_transform(positions, Y)

            ## match points based on distances
            self.match_points(self.positions, Y, positions)
            self.update_positions()

            #print('Total', time.time()-a)
            #print('Threads:', len(self.threads))
    def update_positions(self):
        """
        updates positions based on threads of matched points
        """
        self.positions = np.zeros((len(self.threads), self.threads[0].get_position_mostrecent().shape[0]))

        for i in range(len(self.threads)):
            self.positions[i] = self.threads[i].get_position_mostrecent()


    def match_points(self, existing, new, new_raw):
        """
        matches existing (registered) points to incoming points
        """

        a = time.time()
        #1: Calculate difference matrix
        diff = scipy.spatial.distance.cdist(existing, new,metric='euclidean')

        #print('Dist Mat:', time.time() - a)
        #a = time.time()
        #2: Calculate Matchings
        matchings, unmatched, newpoints = calc_match(diff, self.blob_dist_thresh)
        
        #print('Matchings: ', time.time() - a)
        #a = time.time()

        #3: update matchings
        for match in matchings:
            self.threads[match[0]].update_position(new_raw[match[1]], t = self.t)

        #print('Update:' ,time.time() - a)
        a = time.time()

        #4: create new threads for the new points
        for point in newpoints:
            self.threads.append(Thread(new_raw[point], t=self.t))

        #print('Newthreads:', time.time() - a)
        a = time.time()
        self.t += 1

    def get_positions_t(self, t):
        pass



class Threads:
    def __init__(self, points, im, blob_dist_thresh = 5):
        self.points = np.array(points)
        self.t = []
        for point in points:
            self.t.append(np.array([im[tuple(point)]]))
        self.blob_dist_thresh = blob_dist_thresh
    #def iterate(self, points):
    #    if 

    #def create_thread(self, center):
    #    self.points

    #    pass
    def iterate(self, points):
        TY = self.register(points)
        matching = self.match_points(TY)

    def register(self, points):
        reg = deformable_registration(**{ 'X': points, 'Y': self.points })
        reg.register()
        return reg.TY

    def match_points(self, registered_points, incoming_points):

        _diff = np.zeros((len(incoming_points),len(registered_points)))
        for i in range(len(incoming_points)):
            for j in range(len(registered_points)):
                _diff[i,j] = np.linalg.norm(incoming_points[i]-registered_points[j])
        

        _diff = diff.T

        ### Match current points to registered incoming points ###

        matchings = []
        for i in range(_diff.shape[0]): # iterate over points that already exist 

            if np.min(_diff[i,:]) < self.blob_dist_thresh
                matchings.append([i,np.where(_diff[i,:] == np.min(_diff[i,:]))[0][0]])

                diff[i,:] = 10000
                _diff[:,np.where(_diff[i,:] == np.min(_diff[i,:]))[0][0]] = 10000
            else: pass






        while np.sum(label) != 146:
            xind,yind = np.where(mat == np.min(mat))
            label[xind,yind] += 1 
            diff[i,:] = 10000
            _diff[:,yind] = 10000

    def create_thread()







'''

'''
class minimalThread:
    def __init__(self, position, t_init = 0):
        self.position = np.array(position)
        #self.t = t_init

    def get_position(self):
        return self.position



class Rope:
    def __init__(self, positions, t = 0):
        self.threads = []
        for i in range(len(positions)):
            self.threads.append(Thread(positions[i], t_init = t))


    def 


    def register_newpoints(self, positions):
        reg = deformable_registration(**{ 'X': self.get_positions(), 'Y': positions })
        reg.register

    def get_positions(self):
        _ = []
        for thread in self.threads:
            _.append(thread.get_pos())
        return np.array(_)

    def get_est_positions(self):
        _ = []
        for thread in self.threads:
            _.append(thread.get_pos())
        return np.array(_)



class Thread:
    def __init__(self, position, t_init = 0, p_init = 0.1, q_init = 1, r_init = 0.1):
        """
            Inputs:
                state:     concatenated vector of position and velocity(x,y,z,dxdt,dydt,dzdt)
                t_init:    time of start of thread
                p_init: scaling factor for posterior estimate of covariance matrix
                q_init:    scaling factor for process noise
                r_init: scaling factor for observation noise  
        """

        self.state = np.concatenate(np.array(position),np.array([0,0,0]))
        self.t_init = t_init
        self.t_series = np.zeros(t_init)
        self.dt = 1
        self.F = np.matrix([\
            [1,0,0,self.dt,0,0],\
            [0,1,0,0,self.dt,0],\
            [0,0,1,0,0,self.dt],\
            [0,0,0,1,0,0],\
            [0,0,0,0,1,0],\
            [0,0,0,0,0,1]])
        self.P = p_init*np.identity(6)
        self.Q = q_init*np.identity(6)
        self.R = r_init*np.identity(6)

    def update_F(self):
        self.F = np.matrix([\
            [1,0,0,self.dt,0,0],\
            [0,1,0,0,self.dt,0],\
            [0,0,1,0,0,self.dt],\
            [0,0,0,1,0,0],\
            [0,0,0,0,1,0],\
            [0,0,0,0,0,1]])

    def predict(self):
        self.x_pred = np.dot(F,state)
        self.P = F*P*F.T+Q

    def update(self, obs):
        y_res = obs - self.x_pred
        S = self.P + R
        K = P*np.linalg.inv(S)
        self.x_pred = self.x_pred + np.dot(K, y_res)
        self.P = (np.identity(6)-K)*self.P


    def get_pos(self):
        return self.state[0:3]

    def get_est_pos(self):
        return self.x_pred[0:3]

'''

