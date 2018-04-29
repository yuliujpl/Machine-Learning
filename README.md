# Machine-Learning

Segmentation
------------

To run segmentation analysis, first you must filter dicom images that are not associated with the patient's lung's 3d sliced images. You can do this by running filter_dicom.py:

>> python filter_dicom.py root_dir

Here, root_dir represents the root directory where all your patient folders are stored.

Then you can perform segmentation analysis by running segmentation.py:

>> python segmentation.py root_dir num_workers

root_dir is the same directory used during the filtering step. num_workers is the number of cpus you can allocate on your particular processor for segmentation analysis. Please remmeber that when putting num_workers, you should also consider the amount of memory available in your machine. If your job get's killed during the a particular run, mostly you ran out of memory.

As a rule of thumb, when running patients of 500-1000 slices, it will take up to 8gb of memory per patient. So divide your total memory by 8 and that's the max number of cpu processors you can distribute this segmentation pipeline to.


All segmentation code was adapted from grt123's winning submission for the Kaggle Data Science Bowl 2017. Grt123's code can be found here: https://github.com/lfz/DSB2017.
