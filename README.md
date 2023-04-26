# GCAMP Processing
## Author: FoCo Lab


### Installation Instructions
1. Setup your development environment. If you need help with this, there are many resources on Google. Most people in the lab use Conda as a package manager and develop in Conda virtual environments as well. 

2. Start your virtual environment, and run the command
```bash
pip install git+https://github.com/focolab/eats-worm
```
And that's basically it!

### Usage - calcium activity time series
An example use case is found in `example.py` in the examples directory. The minimal use case is to extract GCaMP timeseries out of a recording. See `example.py` for details.

The only 'coding' necessary here is to modify the 'root' directory that contains all your .tif files from your recording, the number of z-planes used, what frames you want to keep. If you're feeling really fancy, maybe even parameters like the size of your Gaussian filter, the percentile you threshold, the anisotropy/voxel size of your recording, etc. 

### Usage - neuroPAL volumetric images
An example use case is found in `NP_example.py` in the examples directory. The minimal use case is to extract neuron centers out of a neuroPAL recording. See `NP_example.py`

You can make the same edits to the input parameters as in the GCAMP timeseries extraction, with additional options for processing_params and parameters for the NPExtractor object. It will be important to specify processing_params as well as the 'RGBW_channels' (which channels correspond to RGBW pseudocolors in neuroPAL images).

### Slightly More Detailed Explanations 

#### Arguments 

* `root` : path to directory containing your recording
* `numz` : number of z steps in your recording. 
* `offset` : number of frames at the beginning of the recording to throw away. useful if your recording doesn't start at the bottom (i.e. starts on z=5 rather than z=0), or if the scope takes a couple volumes to settle on its final recording position. 
* `frames` : at the NIC, the microscope records a couple frames on its way down. if there are consistently a couple of z steps that you want to keep, specify them here. ex: if there are 10 z-steps in your recording, but you only want to keep z=0,1,2,3,4,5, and 6, write `'frames':[0,1,2,3,4,5,6]`. If not specified, the default is to keep everything. 
* `t` : number of timesteps/volumes to process. this parameter is default set to process your entire recording. only specify this parameter if you want to process a fraction of your recording (i.e. test parameters on the first 100 timesteps or something). 
* `quantile` : first step of the method is to threshold each volume by performing a percentile/quantile threshold before segmentation. a value of 0.99 means that only 1\% of the pixels are kept for analysis. Usually, a value in the range of 0.98 for dimmer recordings or 0.99 for brighter recordings will work well. 
* `gaussian` : a tuple containing the shape of the gaussian filter applied. order goes (width_xy, sigma_xy, width_z, sigma_z). width parameters must be odd. an ad-hoc method for setting these values is: set sigma_xy to be half the diameter of a neuron measured in pixels minus one, set width_xy to 8 times sigma_xy plus one, set width_z to numz/3 and sigma_z to width_z times 3
* `3d` : set to True to perform 3d peak finding, or False for 2d peak finding. as of right now, I don't have a good idea for why to do 3d peak finding over 2d beyond maybe you have less spurios ROIs to deal with.  
* `reg_peak_dist` : regularizes incoming points to prevent oversegmentation. if incoming peaks/neurons are within X pixels of each other, take the peak with the brightest intensity and remove the other. 
* `blob_merge_dist_thresh` : neighborhood (in pixels) to register incoming points to existing thread locations. higher number means that you're doing a search in a broader area. If you're binning your recordings, I would suggest a value of around 4-6. If you're not, maybe around 6-10. also depends on how much motion you expect in your recordings 
* `register_frames` : whether to perform dft image registration in-between each frame. doesn't significantly affect runtime, and is helpful if there is a huge translational shift (worm suddenly moves backwards in channel). 
* `predict` : whether to register new incoming points to predicted positions of existing threads. set to True, if not registering frames, and False if you are registering frames. 
* `regen_mft` : set to False in most cases. set to True if re-initializing an extractor with the exact same parameters. 
* `neuroPAL` : indicates that this is a neuroPAL image and pre-processing should be run.
* `RGBW_channels` : order of RGBW channels in neuroPAL image.
* `resample` : set to True if you want to resample the resolution of the image.
* `old_resolution` : the original resolution of the neuroPAL image in um/pixel. Should be a list in order of [X res, Y res, Z res].
* `new_resolution` : the new resolution you want to resample to.
* `median_filter` : set to True if you want to run a median filtering on the image.
* `median` : set size of median filter kernel. Default is 3x3.
* `histogram_match` : set to True if you want to match the color channel histogram of this image to a reference image. 
* `A_max` : max gray count for your neuroPAL image. e.g. if your acquiring in 12bit, this would be 4096
* `ref_max` : max gray count of the reference image that you're matching the histogram to. e.g. if it's acquired in 16bit, this would be 65536
* `im_to_match` : path to reference image to match the histogram of yours too. 
* `npe_params f` : path to the neuroPAL image tif that you want to show in the NPEX gui. In the above example, it is set to the processed image outputted by the e.process_im() step
* `data_tag` : short description of the dataset you are working with. In the above example, this is 2022-02-12-w01-NP1 which contains the date, number worm, and number image that is being used. For our lab, this is enough information to know exactly where the data comes from.

#### Extractor methods


* `e = Extractor(**arguments)` : initializes the extractor based on arguments above. may take a while to open based on your recording size, since the first thing it does is read through your entire recording to calculate how many frames there are. 
* `e.process_im()` : runs some basic image preprocessing steps based on input arguments in processing_params. Currently supports median filtering, resolution resampling, and histogram matching. For histogram matching, user must supply a path to another file containing reference data to match the histogram to. In this example, we use one of the datasets from the original neuroPAL paper. You should only use this if you are processing single neuroPAL images and not for time series data.
* `e.calc_blob_threads()` : calculates blob threads. also performs automatic border-collision detection and destroys threads that automatically collide with the image border. automatically saves as a numpy pickle object that can be reloaded. 
* `e.quantify()` : performs quantification based on calculated threads. default quantification function is the average of top 10 pixel values in a 7x7 box around the calculated pixel index. automatically saves timeseries in a .txt file that can be read with np.genfromtxt('../timeseries.txt') and re-loaded into a python environment as a numpy array 
* `e.save_MIP()` : saves a maximum intensity projection of your recording in extractor-objects/MIP.tif. note, this only performs a MIP that includes the frames specified in the 'frames' argument. optionally, you can specify a filename or a specific directory to save the MIP. Default is to save it under 'extractor-objects/MIP.tif', but you could pass it 'MIP1.tif' or 'path/to/directory' or 'path/to/directory/MIP.tif' and etc. 
* `e.save_threads()` : saves current state of threads, i.e. if you remove any of them/add any in through code
* `e.save_timeseries()` : saves current state of timeseries, i.e. if you change the timeseries somehow
* `e = load_extractor(root)` : reloads the extractor using the same root directory as specified when it was created. 
* `Curator(e)` : launched an interactive matplotlib GUI that allows you to accept/reject individual blob threads. upon closing of the Python environment, automatically saves your labels as a .json that gets reloaded the next time you run Curator on the loaded extractor. Optionally, it takes an additional argument called "window", which specifies how wide the zoomed-in view of the ROI+neuron goes. There is a known issue that if the ROI is too close to the edge of the image, the red dot in the zoomed in version doesn't necessarily correspond to where the actual found position is. If you suspect that's the case, reduce the window size so that the zoom-in doesn't get affected by image boundaries. 

#### Curator Methods - curating calcium activity time series
* `c = Curator(e)` : creates a Curator object based on an extractor. 
* `c.log_curate()` : force-save the JSON containing which threads you've seen, decided to keep, or decided to trash. This **normally** gets automatcially saved when you quit the Python environment, but in some cases it doesn't happen (not sure why). Just to be save, call this function after every Curator session 
* `c.restart()` : if you're running this in an interactive Python session (line-by-line, or in an IDE), and you have already run `c = Curator(e)` but closed the window and want to resume where you've left off, run `c.restart()` to re-open the matplotlib window and pick up curating where you left off. 

#### Curator Methods - curating neuroPAL volumetric image 

* `npe = npex.NPExtractor(**npe_params)` : creates a NPExtractor object based on input parameters. Will access files created in the Extractor output folder.
* `npe.launch_gui()` : opens up the NPEX GUI where the user can curate detected neuron centers. Users can confirm, delete, or add neuron centers in this view. Make sure to click export blobs in the GUI to export the results of your curation.
* `npe.export()` :  exports relevant metadata from NPExtractor object to json in output folder. 

Generally, you will follow the sequence of:
```python3
from eats_worm import *
arguments = {
...
}

e = Extractor(**arguments)
e.calc_blob_threads()
e.quantify()
c = Curator(e)
c.log_curate()
```

Or, if you are working with neuroPAL volumetric images:

```python3
from eats_worm import *
from npex import npex
eats_params = {
...
}

npe_params = {
...
}

e = Extractor(**eats_params)
e.process_im()
e.calc_blob_threads()
e.quantify()

npe = npex.NPExtractor(**npe_params)
npe.launch_gui()
npe.export()
```


If you're for some reason doing processing across multiple sessions, you can reload the extractor with


```python3
from eats_worm import *
arguments = {
...
}

e = load_extractor(arguments['root'])
...
...
```

If something goes wrong during reloading, you usually fix it by deleting the associated object in the extractor-objects folder. i.e. if your MultiFileTiff is behaving strangely, delete mft.obj. If you're curating something, and the labels aren't behaving like they should, delete or move curate.json. 




Known Issues:
- Matplotlib interactive GUI in Curator doesn't work in some IDEs (e.g. Spyder). If possible, write a script to perform your processing/curating and run it from command line/terminal/conda terminal







