
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
    
    def infill(self, parents=None, weights=None):
        for i in range(len(self.threads)):
            if self.threads[i].t[0]==0:
                pass
            else:
                inferred = self.threads[i].get_position_t(self.threads[i].t[0])
                for j in reversed(range(self.threads[i].t[0])):
                    if parents:
                        if not weights:
                            weights = np.array([1] * len(parents))
                        parent_offsets = np.array([self.threads[parent].get_position_t(j + 1) - self.threads[parent].get_position_t(j) for parent, weight in zip(parents, weights)])
                        inferred -= np.sum(parent_offsets, axis=0) / np.sum(weights)
                    else:
                        inferred = inferred - self.dvec[j]
                    self.threads[i].infill(inferred)

    def exfill(self, parents=None, weights=None):
        for i in range(len(self.threads)):
            if self.threads[i].t[-1]==self.maxt - 1:
                pass
            else:
                inferred = self.threads[i].get_position_t(self.threads[i].t[-1])
                for j in range(self.threads[i].t[-1] + 1, self.maxt):
                    if parents:
                        if not weights:
                            weights = np.array([1] * len(parents))
                        parent_offsets = np.array([self.threads[parent].get_position_t(j) - self.threads[parent].get_position_t(j - 1) for parent, weight in zip(parents, weights)])
                        inferred += np.sum(parent_offsets, axis=0) / np.sum(weights)
                    else:
                        inferred = inferred + self.dvec[j-1]
                    self.threads[i].exfill(inferred)

    # handle threads which are illegally close to one another (e.g. after infill)
    def manage_collisions(self, method='merge', anisotropy=None):
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
                    if anisotropy:
                        t_positions *= anisotropy
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

    def alter_thread_post_hoc(self, thread, position_0, position_1, time_0, time_1):
        print("Interpolating positions for thread", thread, "between timepoints", time_0, "and", time_1)
        pos_diff = position_1 - position_0
        time_diff = time_1 - time_0
        pos = position_0
        for t in range(time_0, time_1 + 1):
            self.threads[thread].positions[t] = position_0 + ((t - time_0 + 1) / time_diff) * pos_diff
        self.update_positions()
        self.make_allthreads()
        return True

    # handle manual addition of new roi to completed spool
    def add_thread_post_hoc(self, position, t, anisotropy, excluded_threads=None):
        distances = scipy.spatial.distance.cdist(np.array([position]), anisotropy * self.get_positions_t(t), metric='euclidean')
        if excluded_threads:
            for thread in excluded_threads:
                distances[:,thread] = np.Inf
        if np.min(distances) < 1:
            print("Invalid creation of new ROI on top of existing ROI; ignoring.")
            return False
        num_neighbors = np.minimum(3, len(distances) - 1)
        nearest_neighbors = np.argpartition(distances, num_neighbors)[:num_neighbors]
        self.threads.append(Thread(position, t=t, maxt = self.maxt))
        self.infill(parents=nearest_neighbors, weights=1./distances[nearest_neighbors])
        self.exfill(parents=nearest_neighbors, weights=1./distances[nearest_neighbors])
        self.update_positions()
        self.make_allthreads()
        return True

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
            df['ID'] = [th.label if th.label is not None else ""]*len(df)
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
