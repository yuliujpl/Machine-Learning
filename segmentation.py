
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os, sys
import scipy.ndimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from skimage import measure, morphology
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
from scipy.io import loadmat
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial


data_path = sys.argv[1]
PATH = sys.argv[1]
output_path = working_path = sys.argv[2]

#get_ipython().magic('matplotlib inline')


# Print out the first 5 file names to verify we're in the right folder.



def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)





#patient_0_image, patient_0_spacing = get_pixels_hu(load_scan(patient_0_path))
#spacing = patient_0_spacing
def plot_3d(image, label, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    print ("Small")
    p = image.transpose(2,1,0)
    p = measure.block_reduce(p,(3,3,3), func=np.mean)
    print ("Med")
    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    print ("Large")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    print ("Super")
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    print ("Super2")
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    print ("Super3")
    ax.add_collection3d(mesh)

    print ("Super4")
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    print ("Yeah")
    plt.savefig(label, bbox_inches='tight')
    
#plot_3d(patient_0_image, 400)




#bw = np.zeros(patient_0_image.shape, dtype=bool)
#plt.imshow(patient_0_image[0], cmap=plt.cm.gray) #show slices with background blacked out
#grid_axis = np.linspace(-image_size/2 + 0.5 , image_size/2. - 0.5, image_size)
#x, y = np.meshgrid(grid_axis, grid_axis)
#d = (x**2 + y**2)**0.5
#nan_mask = (d < image_size/2).astype(float)
#nan_mask[nan_mask == 0] = np.nan
#plt.imshow(patient_0_image[300], cmap=plt.cm.gray) #show image before gaussian filter to remove noise
#plt.imshow(scipy.ndimage.filters.gaussian_filter(patient_0_image[300], sigma=1, truncate=2.0), cmap=plt.cm.gray) #show image after gaussian filter to remove noise

#if len(np.unique(patient_0_image[300,:10,:10])) == 1:
#    current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(patient_0_image[300],nan_mask), sigma=1, truncate=2.0) < -600
#else:
#    current_bw = scipy.ndimage.filters.gaussian_filter(patient_0_image[300], sigma=1, truncate=2.0) < -600
#plt.imshow(current_bw.astype(float), cmap=plt.cm.gray) #generate mask for a slice

#label = measure.label(current_bw)
#plt.imshow(label, cmap=plt.cm.gray)

#properties = measure.regionprops(label)
#valid_label = set()
#for prop in properties:
#    if prop.area * spacing[1]*spacing[2] > 30 and prop.eccentricity < 0.99:
#        valid_label.add(prop.label)
        
#current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
#plt.imshow(current_bw, cmap=plt.cm.gray)

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw
    
#patient_0_binary_image = binarize_per_slice(patient_0_image, spacing)
#plot_3d(patient_0_binary_image.astype(int), threshold=0)

    
#image_size = patient_0_image.shape[1]
#image_size
#plt.imshow(nan_mask, cmap=plt.cm.gray) #check distance from origin




def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    print("all-slice")
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1],                     label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1],                     label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    if len(np.unique(label)) == 1:
        return bw, 0
    
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    if len(valid_label) == 0:
        return bw, 0
    
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

#bw = patient_0_binary_image
#flag = 0
#cut_num = 0
#cut_step = 2
#bw0 = np.copy(bw)
#while flag == 0 and cut_num < bw.shape[0]:
#    print(cut_num)
#    bw = np.copy(bw0)
#    bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
#    cut_num = cut_num + cut_step
    
#plot_3d(bw.astype(int), 0) #Plut trimed lung





def fill_hole(bw):
    label = measure.label(~bw)
    bg_labels = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1],                      label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_labels)).reshape(label.shape)
    
    return bw

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw
def step1_python(case_path, prep_folder):
    print ("Step1 Start")
    case = load_scan(case_path)
    print ("Loaded")
    case_pixels, spacing = get_pixels_hu(case)
    print ("Hu pixels")
    plot_3d(case_pixels, os.path.join(prep_folder,"1_patient_structure.png"), 400)
    bw = binarize_per_slice(case_pixels, spacing)
    print ("BW")
    #print (bw.astype(int))
    #print (bw)
    plot_3d(bw.astype(int), os.path.join(prep_folder,"2_pre_segment.png"), 0)
    print ("Pre")
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    print("step1")
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    print (bw1)
    print (bw2)
    plot_3d(bw1 | bw2, os.path.join(prep_folder,"3_post_segment.png"), 0)
    return case_pixels, bw1, bw2, spacing
def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg
def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
        
def savenpy(id,filelist,prep_folder,data_path,use_existing=True):      
    print("Starting {}".format(id))
    resolution = np.array([1,1,1])
    name = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
    try:
        print("Running")
        im, m1, m2, spacing = step1_python(os.path.join(data_path,name), prep_folder)
        Mask = m1+m2
        
        print("New shape")
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')



        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        plot_3d(bones.astype(int), os.path.join(prep_folder,"4_bones_only.png"), 0)
        sliceim[bones] = pad_value
        plot_3d((~(sliceim==pad_value)).astype(int), os.path.join(prep_folder,"5_remove_bones.png"), 0)
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        plot_3d((~(sliceim2==pad_value)).astype(int), os.path.join(prep_folder,"6_final_preprocessed.png"), 0)
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
        np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]]))
    except:
        print('bug in '+name)
        raise
    print(name+' done')

def full_prep(data_path,prep_folder,n_worker = None,use_existing=True):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

            
    print('starting preprocessing')
    #pool = Pool(n_worker)
    filelist = [f for f in os.listdir(data_path)]
    partial_savenpy = partial(savenpy,filelist=filelist,prep_folder=prep_folder,
                              data_path=data_path,use_existing=use_existing)

    print ("Looping")
    N = len(filelist)
    for idx in range(0,N):
      partial_savenpy(idx)
    #_=pool.map(partial_savenpy,range(N))
    #pool.close()
    #pool.join()
    print('end preprocessing')
    return filelist

print("FULL PREP")
full_prep(data_path,working_path,n_worker=1)











