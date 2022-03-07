
import os
import pickle
import pandas as pd
import numpy as np
import pdb
import time
import scipy.spatial
from scipy.optimize import linear_sum_assignment
import copy

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
        self.dvec = np.zeros((self.maxt-1,3))
        self.allthreads = None

    def export(self, f):
        print('Saving spool as pickle object...')
        os.makedirs(os.path.dirname(f), exist_ok=True)
        file_pi = open(f, 'wb')
        pickle.dump(self, file_pi)
        file_pi.close()

    @classmethod
    def load(cls, f):
        if not f:
            raise Exception('pickle file (f) is required')
        print('Loading spool from pickle object...')
        with open(f,'rb') as fopen:
            x = pickle.load(fopen)
        return x

    def reel(self, positions, anisotropy = (6,1,1), delta_t=1, offset=np.array([0,0,0])):

        # if no threads already exist, add all incoming points to new threads them and positions tracker
        if self.threads == []:
            for i in range(len(positions)):
                self.threads.append(Thread(positions[i], t=delta_t-1, maxt = self.maxt))

            # update numpy array containing most recently found position of all blobs
            self.update_positions()

            # update numpy array containing predictions of positions for the next incoming timepoint
            self.predictions = copy.copy(self.positions)
            self.t = delta_t

        # if threads already exist
        else:
            # match points based on a max-threshold euclidean distance based matching
            #try:

            # if doing dft registration, there's an offset in the arguments for reel. offset predictions by that amount
            self.predictions = self.predictions - offset

            # if doing some anisotropy processing, need to offset positions by the anisotropy factor
            for i in range(len(anisotropy)):
                self.predictions[:,i]=self.predictions[:,i]*anisotropy[i]
                positions[:,i]=positions[:,i]*anisotropy[i]

            # calculate distance matrix to perform matching on
            diff = scipy.spatial.distance.cdist(self.predictions, positions, metric='euclidean')
            
            # reset the positions to their original coordinate (pixel coordinates)
            for i in range(len(anisotropy)):        
                self.predictions[:,i]=self.predictions[:,i]/anisotropy[i]
                positions[:,i]=positions[:,i]/anisotropy[i]
            
            # calculate the matchings 
            matchings, unmatched, newpoints = self.calc_match(diff, self.blob_dist_thresh)

            #dvec[self.t] = np.zeros(self.threads[0].get_position_mostrecent().shape)
            
            # for all the incoming peaks that were matched to existing threads
            for match in matchings:

                # update dvec
                interpolated = (self.threads[match[0]].get_position_mostrecent()-positions[match[1]]) / delta_t
                for t in range(delta_t):
                    self.dvec[self.t - 1 + t] -= interpolated
                    self.threads[match[0]].update_position(positions[match[1]] + interpolated * (delta_t - t), t=self.t+t, found = True)
            if matchings.any():
                for t in range(delta_t):
                    self.dvec[self.t - 1 + t] *= 1/len(matchings)
            else:
                for t in range(delta_t):
                    self.dvec[self.t - 1 + t] = 0
            #print(self.dvec[self.t])
            for match in unmatched:
                for t in range(delta_t):
                    self.threads[match].update_position(self.threads[match].get_position_mostrecent() + self.dvec[self.t - 1 + t], found = False, t=self.t+t)

            for point in newpoints:
                self.threads.append(Thread(positions[point], t=self.t + delta_t - 1, maxt = self.maxt))

            self.update_positions()
            self.update_predictions()
            self.t += delta_t

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
        mat_copy = np.copy(mat)
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
            # if new point is too close to any old point, don't add!
            bad_newpoints = []
            for point in newpoints:
                if np.min(mat_copy[:,point]) < thresh:
                    bad_newpoints.append(point)
            for bad_newpoint in bad_newpoints:
                newpoints.remove(bad_newpoint)
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
                self.predictions[i] = self.threads[i].get_position_mostrecent() + self.dvec[self.t-1]
                
                if len(self.threads[i].t) > 1:
                    self.predictions[i] = self.threads[i].get_position_t(self.threads[i].t[-1])+ self.dvec[self.t-1]
                else:
                    self.predictions[i] = self.threads[i].get_position_mostrecent() + self.dvec[self.t-1]
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

    def exfill(self):
        for i in range(len(self.threads)):
            if self.threads[i].t[-1]==self.maxt - 1:
                pass
            else:
                inferred = self.threads[i].get_position_t(self.threads[i].t[-1])
                for j in range(self.threads[i].t[-1] + 1, self.maxt):
                    inferred = inferred + self.dvec[j-1]
                    self.threads[i].exfill(inferred)

    # handle threads which are illegally close to one another (e.g. after infill)
    def manage_collisions(self, method='merge'):
        if method is None:
            pass

        # greedily prune illegally close threads, keeping the older thread
        elif method == 'prune':
            if self.allthreads is not None:
                threads_collided = set()
                threads_to_remove = set()
                for t in range(self.maxt):
                    t_positions = self.allthreads[t]
                    t_positions = t_positions.reshape((-1, 3))
                    distances = scipy.spatial.distance.cdist(t_positions, t_positions, metric='euclidean')
                    # zero out diagonal and below to avoid identities and duplicates
                    tril_mask = np.tril(np.ones(distances.shape, dtype=bool))
                    distances[tril_mask] = self.blob_dist_thresh + 1
                    for removed_thread in threads_to_remove:
                        distances[:,removed_thread] = self.blob_dist_thresh + 1
                        distances[removed_thread, :] = self.blob_dist_thresh + 1
                    collided = np.argwhere(distances < self.blob_dist_thresh)
                    sorted_collided = collided[(np.argsort(distances[tuple(collided.T)], axis=None).T)]

                    for collision in sorted_collided:
                        if distances[tuple(collision)] < self.blob_dist_thresh:
                            earlier_thread, later_thread = sorted(collision)
                            threads_collided.update([earlier_thread, later_thread])
                            threads_to_remove.add(later_thread)
                            distances[:,later_thread] = self.blob_dist_thresh + 1
                            distances[later_thread, :] = self.blob_dist_thresh + 1

                for i in sorted(list(threads_to_remove), reverse=True):
                    self.threads.pop(i)
                print('Blob threads collided:', len(threads_collided), 'of', self.allthreads[t].reshape((-1, 3)).shape[0], '. Pruned to ', len(threads_collided) - len(threads_to_remove), 'distinct threads.')
                self.update_positions()
                self.make_allthreads()
            else:
                print('Not managing collisions. make_allthreads() must be called before managing collisions.')

        # merge illegally close threads. too much merging right now; needs to be updated to be iterative
        elif method == 'merge':
            if self.allthreads is not None:
                collisions = []
                for t in range(self.maxt):
                    t_positions = self.allthreads[t]
                    t_positions = t_positions.reshape((-1, 3))
                    distances = scipy.spatial.distance.cdist(t_positions, t_positions, metric='euclidean')
                    # zero out diagonal and below to avoid identities and duplicates
                    tril_mask = np.tril(np.ones(distances.shape, dtype=bool))
                    distances[tril_mask] = self.blob_dist_thresh + 1
                    collided = np.argwhere(distances < self.blob_dist_thresh)

                    for collision in collided:
                        first_roi_group, second_roi_group = None, None
                        for collision_group in collisions:
                            if collision[0] in collision_group:
                                first_roi_group = collision_group
                            if collision[1] in collision_group:
                                second_roi_group = collision_group
                        if first_roi_group and second_roi_group and first_roi_group != second_roi_group:
                            first_roi_group |= second_roi_group
                            collisions.remove(second_roi_group)
                        elif first_roi_group:
                            first_roi_group.add(collision[1])
                        elif second_roi_group:
                            second_roi_group.add(collision[0])
                        else:
                            collisions.append(set(collision.tolist()))

                threads_to_remove = set()
                for collision_group in collisions:
                    collision_group = sorted(list(collision_group))
                    for t in range(self.maxt):
                        position = self.threads[collision_group[0]].positions[t]
                        for thread in collision_group[1:]:
                            position += self.threads[thread].positions[t]
                            threads_to_remove.add(thread)
                        position /= len(collision_group)
                        self.threads[collision_group[0]].positions[t] = position

                for i in sorted(list(threads_to_remove), reverse=True):
                    self.threads.pop(i)
                print('Blob threads collided:', len(collisions) + len(threads_to_remove), 'of', self.allthreads[t].reshape((-1, 3)).shape[0], '. Merged to ', len(collisions), 'distinct threads.')
                self.update_positions()
                self.make_allthreads()
            else:
                print('Not managing collisions. make_allthreads() must be called before managing collisions.')

    def make_allthreads(self):
        # initialize numpy array based on how many timepoints and number of threads
        self.allthreads = np.zeros((self.maxt, 3*len(self.threads)))

        # fill in everything
        for i in range(len(self.threads)):
            self.allthreads[:,3*i:3*i+3] = self.threads[i].positions

    # handle manual addition of new roi to completed spool
    def add_thread_post_hoc(self, position, t):
        self.threads.append(Thread(position, t=t, maxt = self.maxt))
        self.infill()
        self.exfill()
        self.update_positions()
        self.make_allthreads()

    def get_positions_t(self,t,indices=None):
        if self.allthreads is not None:

            t = int(t)
            
            if t >= self.maxt:
                return False
            elif t < 0: return False
            if indices is None:
                return self.allthreads[t].reshape((-1,3))
            return self.allthreads[t].reshape((-1,3))[indices]

        else:
            print('Run make_allthreads first')
            return False

    def get_positions_t_z(self,t,z, indices=None):
        # get positions first
        _a = self.get_positions_t(t, indices)

        z = int(z)
        
        
        return _a[np.rint(_a[:,0])==z]

    def to_dataframe(self, dims):
        """package results to a dataframe

        parameters
        ----------
        dims (list): Required to specify dimension order e.g. ['Z', 'Y', 'X']

        returns
        -------
        df_out (pandas.DataFrame):
        """
        dd = {True:'detected', False:'infilled'}

        all_dataframes = []
        for ix, th in enumerate(self.threads):
            df = pd.DataFrame(data=th.positions, columns=dims)
            df['T'] = th.t
            df['prov'] = [dd[k] for k in th.found]
            df['blob_ix'] = [ix]*len(df)
            all_dataframes.append(df)
        df_out = pd.concat(all_dataframes, axis=0).reset_index(drop=True)
        return df_out


class Thread:
    """
    Class for single blob thread. Contains the following 
    Properties:
        - positions:     list of positions that the current blob was found at, indexed the same as image indexing
        - t:             list of time points that the current blob was found at, i.e. position[i] was found at time point t[i]

    Methods:
        - get_position_mostrecent():    returns most recent position
        - update_position(position, t): updates list of positions and time point; default t is most recent t + 1
        - get_positions_t(t):           returns position at time point specified, and if blob wasn't found at that time, then returns the position at the most recent time before time point specified

    Most recent edit: 
    10/23/2019
    """
    def __init__(self, position = [], t = 0, **kwargs):
        maxt = kwargs.get('maxt')
        self.positions = np.zeros((maxt,3))
        self.found = np.zeros((maxt))
        #self.positions = []
        self.t = []
        if position != []:
            self.positions[t] = np.array(position)
            #self.t = t + 1
            #self.positions.append(np.array(position))
            self.t.append(int(t))
        self.label = kwargs.get('label')

    def get_position_mostrecent(self):
        """    
        returns most recent position
        """
        return self.positions[self.t[-1]]

    def update_position(self, position, found = False, t = 0):
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

        if found:
            self.found[t] = True

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

    def exfill(self, position):
        self.t.append(self.t[-1]+1)
        self.positions[self.t[-1]] = np.array(position)

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

