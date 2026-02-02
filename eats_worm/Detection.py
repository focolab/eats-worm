import numpy as np
from improc.segfunctions import *
from improc.segfunctions import peak_local_max as ani_peak_local_max
from scipy.ndimage import rotate
from skimage.registration import phase_cross_correlation
from skimage.filters import rank
from skimage.feature import peak_local_max
import cv2

class PeaksResult:
    def __init__(self, positions=None, delta_t = 1, offset = np.array([0,0,0])):
        self.positions = positions
        self.delta_t = delta_t
        self.offset = offset


class Segmenter(object):
    def __new__(cls, algorithm, *args, **kwargs):
        tmip_options = ['tmip', 'tmip_2d_template', 'seeded_tmip', 'seeded_tmip_2d_template']
        fbf_options = ['skimage', 'threed', 'template', 'curated', 'seeded_skimage', 'findpeaks2d']

        valid_options = tmip_options + fbf_options

        if cls is Segmenter:
            if algorithm in tmip_options : return super(Segmenter, cls).__new__(TMIPSegmenter)
            if algorithm in fbf_options : return super(Segmenter, cls).__new__(FrameByFrameSegmenter)

            raise ValueError(f"Unknown algorithm {algorithm}. Options are: {valid_options}")
        else:
            return super(Segmenter, cls).__new__(cls)

    def __init__(self, algorithm, im, algorithm_params):
        self.algorithm = algorithm
        self.im = im
        self.algorithm_params = algorithm_params

        self.frames = self.im.frames

        ## peakfinding/spooling parameters
        self.gaussian = algorithm_params.get('gaussian', (25,4,3,1))
        self.median = algorithm_params.get('median', 3)
        self.quantile = algorithm_params.get('quantile', 0.99)
        self.reg_peak_dist = algorithm_params.get('reg_peak_dist', 40)
        self.peakfind_channel = algorithm_params.get('peakfind_channel',0)
        self.register = algorithm_params.get('register', False)
        self.curator_layers = algorithm_params.get('curator_layers', False)
        try:
            self.algorithm_params['templates_made'] = type(self.algorithm_params['template']) != bool
        except:
            self.algorithm_params['templates_made'] = False

        # self.start_t and self.end_t are time index cutoffs for partial analysis
        self.start_t = algorithm_params['start_t']
        self.end_t = algorithm_params['end_t']


    def process_step(self):
        """
        Subclasses must implement this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement 'process_step'")


class TMIPSegmenter(Segmenter):
    def __init__(self, algorithm, im, algorithm_params):
        super().__init__(algorithm, im, algorithm_params)
        self.window_size = self.algorithm_params.get('window_size', 10)
        self.ims = []
        self.offsets = []
        self.last_offset = None
        self.last_im = None
        self.templated = '2d_template' in self.algorithm
        self.seeded = 'seeded' in self.algorithm
        if self.templated and self.curator_layers:
            self.curator_layers = {'filtered': {'type': 'image', 'data' : []}}

        if all(isinstance(dimension, int) for dimension in self.im.anisotropy):
            expanded_shape = tuple([dim_len * ani for dim_len, ani in zip(self.im.get_t(0).shape, self.im.anisotropy)])
            self.mask = np.zeros(expanded_shape, dtype=np.uint16)
            self.mask[tuple([np.s_[::ani] for ani in self.im.anisotropy])] = 1


    
    def process_step(self, t):
        im1 = self.im.get_t(t, channel = self.peakfind_channel)
        im1 = medFilter2d(im1, self.median)

        if self.seeded and t == self.start_t:
            peaks = np.array(self.algorithm_params['peaks'])
            self.last_offset = np.array([0,0,0])
            self.last_im = np.copy(im1)
            return PeaksResult(peaks)
        
        else:
            if self.register and t != self.start_t:
                _off = phase_cross_correlation(self.last_im, im1, upsample_factor=100)[0][1:]
                _off = np.insert(_off, 0, 0)
                # track offset relative to t=0 so that all tmips are spatially aligned
                _off += self.last_offset
                self.offsets.append(_off)
                shift = tuple(np.rint(_off).astype(int))
                self.last_im = np.copy(im1)
                im1 = shift3d(im1, shift)
            else:
                self.last_im = np.copy(im1)
                self.offsets.append(np.array([0, 0, 0]))

            self.last_offset = self.offsets[-1]
            self.ims.append(np.copy(im1))

            if len(self.ims) == self.window_size or t == self.end_t - 1:
                tmip = np.max(np.array(self.ims), axis=0)

                if self.templated:
                    vol_tmip = np.max(np.array(self.ims), axis=0).astype(np.float32)

                    for z in range(vol_tmip.shape[0]):
                        im_filtered = vol_tmip[z, :, :]
                        fb_threshold_margin = self.algorithm_params.get('fb_threshold_margin', 20)
                        threshold = np.median(im_filtered) + fb_threshold_margin
                        im_filtered = (im_filtered > threshold) * im_filtered

                        gaussian_template_filter = render_gaussian_2d(self.algorithm_params.get('gaussian_diameter', 13), self.algorithm_params.get('gaussian_sigma', 9))
                        im_filtered = cv2.matchTemplate(im_filtered, gaussian_template_filter, cv2.TM_CCOEFF)
                        pad = [int((x-1)/2) for x in gaussian_template_filter.shape]
                        im_filtered = np.pad(im_filtered, tuple(zip(pad, pad)))

                        im_filtered = (im_filtered > threshold) * im_filtered
                        vol_tmip[z, :, :] = im_filtered
                    
                    if self.curator_layers:
                        vol_tmip_mask_as_list = (vol_tmip > 0).tolist()
                        for timepoint in range(len(self.ims)):
                            self.curator_layers['filtered']['data'].append(vol_tmip_mask_as_list)
                    
                    # reset local var
                    tmip = vol_tmip

                    # get peaks, min distance is in 3 dimensions
                    if all(isinstance(dimension, int) for dimension in self.im.anisotropy):
                        expanded_im = np.repeat(tmip, self.im.anisotropy[0], axis=0)
                        expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                        expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                        expanded_im *= self.mask
                        peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), exclude_border = self.algorithm_params.get('exclude_border', True)).astype(float)
                        peaks /= self.im.anisotropy
                    else:
                        peaks = ani_peak_local_max(tmip, min_distance=self.algorithm_params.get('min_distance', 9), exclude_border=self.algorithm_params.get('exclude_border', True), anisotropy=self.im.anisotropy).astype(float)
                else:
                    tmip = np.max(np.array(self.ims), axis=0)
                    expanded_im = np.repeat(tmip, self.im.anisotropy[0], axis=0)
                    expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                    expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                    expanded_im *= self.mask
                    peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), num_peaks=self.algorithm_params.get('num_peaks', 50)).astype(float)
                    peaks /= self.im.anisotropy
                
                delta_t = len(self.ims)
                self.ims = []
                self.offsets = []
                return PeaksResult(peaks - self.last_offset, delta_t=delta_t)

            else:
                # if not made TMIP, return empty result
                return PeaksResult()




    



class FrameByFrameSegmenter(Segmenter):
    def __init__(self, algorithm, im, algorithm_params):
        super().__init__(algorithm, im, algorithm_params)
        if algorithm in ['skimage', 'template']:
            if all(isinstance(dimension, int) for dimension in self.im.anisotropy):
                expanded_shape = tuple([dim_len * ani for dim_len, ani in zip(self.im.get_t(0).shape, self.im.anisotropy)])
                self.mask = np.zeros(expanded_shape, dtype=np.uint16)
                self.mask[tuple([np.s_[::ani] for ani in self.im.anisotropy])] = 1

    def process_step(self, t):
        im1 = self.im.get_t(t, channel = self.peakfind_channel)
        im1 = medFilter2d(im1, self.median)
        if self.gaussian:
            im1 = gaussian3d(im1, self.gaussian)

        im1 = np.array(im1 * np.array(im1 > np.quantile(im1, self.quantile)))


        if all(isinstance(dimension, int) for dimension in self.im.anisotropy):
            expanded_shape = tuple([dim_len * ani for dim_len, ani in zip(self.im.get_t(0).shape, self.im.anisotropy)])
            mask = np.zeros(expanded_shape, dtype=np.uint16)
            mask[tuple([np.s_[::ani] for ani in self.im.anisotropy])] = 1

        if self.algorithm == 'skimage':
            expanded_im = np.repeat(im1, self.im.anisotropy[0], axis=0)
            expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
            expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
            expanded_im *= self.mask
            peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), num_peaks=self.algorithm_params.get('num_peaks', 50))
            peaks //= self.im.anisotropy
            min_filter_size = self.algorithm_params.get('min_filter', False)
            if min_filter_size:
                min_filtered = rank.minimum(im1.astype(bool), np.ones((1, min_filter_size, min_filter_size)))
                peaks_mask = np.zeros(im1.shape, dtype=bool)
                peaks_mask[tuple(peaks.T)] = True
                peaks = np.array(np.nonzero(min_filtered * peaks_mask)).T

        elif self.algorithm == 'threed':
            peaks = findpeaks3d(im1)
            peaks = reg_peaks(im1, peaks, thresh=self.reg_peak_dist)
            
        elif self.algorithm == 'template':
            if not self.algorithm_params['templates_made']:
                expanded_im = np.repeat(im1, self.im.anisotropy[0], axis=0)
                expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                expanded_im *= self.mask
                try:
                    peaks = np.rint(self.algorithm_params["template_peaks"]).astype(int)
                except:
                    peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), num_peaks=self.algorithm_params.get('num_peaks', 50))
                    peaks //= self.im.anisotropy
                chunks = get_bounded_chunks(data=im1, peaks=peaks, pad=[1, 25, 25])
                chunk_shapes = [chunk.shape for chunk in chunks]
                max_chunk_shape = (max([chunk_shape[0] for chunk_shape in chunk_shapes]), max([chunk_shape[1] for chunk_shape in chunk_shapes]), max([chunk_shape[2] for chunk_shape in chunk_shapes]))
                self.templates = [np.mean(np.array([chunk for chunk in chunks if chunk.shape == max_chunk_shape]), axis=0)]
                quantiles = self.algorithm_params.get('quantiles', [0.5])
                rotations = self.algorithm_params.get('rotations', [0])
                for quantile in quantiles:
                    for rotation in rotations:
                        try:
                            self.templates.append(rotate(np.quantile(chunks, quantile, axis=0), rotation, axes=(-1, -2)))
                        except:
                            pass
                print("Total number of computed templates: ", len(self.templates))
                self.algorithm_params['templates_made'] = True
            peaks=None
            for template in self.templates:
                template_peaks = peak_filter_2(data=im1, params={'template': template, 'threshold': 0.5})
                if peaks is None:
                    peaks = template_peaks
                else:
                    peaks = np.concatenate((peaks, template_peaks))
            peak_mask = np.zeros(im1.shape, dtype=bool)
            peak_mask[tuple(peaks.T)] = True
            peak_masked = im1 * peak_mask
            expanded_im = np.repeat(peak_masked, self.im.anisotropy[0], axis=0)
            expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
            expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
            expanded_im *= self.mask
            peaks = peak_local_max(expanded_im, min_distance=13)
            peaks //= self.im.anisotropy
        
        elif self.algorithm == 'curated':
            if t == 0:
                peaks = np.array(self.algorithm_params['peaks'])
                self.last_peaks = peaks
            else:
                peaks = self.last_peaks
            
        elif algorithm == 'seeded_skimage':
            if t == 0:
                peaks = np.array(self.algorithm_params['peaks'])
            else:
                expanded_im = np.repeat(im1, self.im.anisotropy[0], axis=0)
                expanded_im = np.repeat(expanded_im, self.im.anisotropy[1], axis=1)
                expanded_im = np.repeat(expanded_im, self.im.anisotropy[2], axis=2)
                expanded_im *= self.mask
                peaks = peak_local_max(expanded_im, min_distance=self.algorithm_params.get('min_distance', 9), num_peaks=self.algorithm_params.get('num_peaks', 50))
                peaks //= self.im.anisotropy

        if self.register and t != self.start_t:
            _off = phase_cross_correlation(self.last_im1, im1, upsample_factor=100)[0][1:]
            _off = np.insert(_off, 0,0)
            if self.algorithm == 'curated':
                self.last_peaks -= _off
                return PeaksResult(peaks, offset=_off)

        self.last_im1 = np.copy(im1)
        return PeaksResult(peaks)

        

        
        





def shift3d(im, shift):
    if np.max(np.abs(shift)) > 0:
        axis = tuple(np.arange(im.ndim))
        im = np.roll(im, shift, axis)
        if shift[0] >= 0:
            im[:shift[0], :, :] = 0
        else:
            im[shift[0]:, :, :] = 0
        if shift[1] >= 0:
            im[:, :shift[1], :] = 0
        else:
            im[:, shift[1]:, :] = 0
        if shift[2] >= 0:
            im[:, :, :shift[2]] = 0
        else:
            im[:, :, shift[2]:] = 0
    return im


def render_gaussian_2d(blob_diameter, sigma):
    """
    :param im_width:
    :param sigma:
    :return:
    """

    gaussian = np.zeros((blob_diameter, blob_diameter), dtype=np.float32)

    # gaussian filter
    for i in range(int(-(blob_diameter - 1) / 2), int((blob_diameter + 1) / 2)):
        for j in range(int(-(blob_diameter - 1) / 2), int((blob_diameter + 1) / 2)):
            x0 = int((blob_diameter) / 2)  # center
            y0 = int((blob_diameter) / 2)  # center
            x = i + x0  # row
            y = j + y0  # col
            gaussian[y, x] = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2 / sigma / sigma)

    return gaussian