# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:49:22 2014

@author: alberts,tetteh
"""
from __future__ import print_function
import os as os
import SimpleITK as itk
import numpy as np

def make_itk_image(imageArray,protoImage=None):
    ''' Create an itk image given an image numpy ndarray (imageArray) and an
    itk proto-image (protoImage) to provide Origin, Spacing and Direction.'''

    image = itk.GetImageFromArray(imageArray)
    if protoImage != None:
        image.CopyInformation(protoImage)

    return image

def make_itk_image_series(imageArray4D,protoImage):
    ''' Create a 4D itk image given a 4-dimensional image numpy ndarray
    (imageArray4D) and a 3-dimensional itk proto-image (protoImage) for the
    3D volumes within the 4D array.'''

    if len(imageArray4D.shape) is not 4:
        print('This function is written for 4dimensional input arrays only')
        return

    mainDim = imageArray4D.shape[0]
    imageList = []
    for dim in range(mainDim):
        imageArray = imageArray4D[dim,:]
        image = make_itk_image(imageArray,protoImage)
        imageList.append(image)

    image4D = itk.JoinSeries(*imageList)

    return image4D

def write_itk_imageArray(imageArray,filename):
    img = make_itk_image(imageArray)
    write_itk_image(img,filename)

def write_itk_image(image,filename):
    ''' Write an itk image to a specified filename.'''

    writer = itk.ImageFileWriter()
    writer.SetFileName(filename)

    if filename.endswith('.nii'):
        Warning('You are converting nii, be careful with type conversions')

    writer.Execute(image)

    return

def get_itk_image(filename):
    ''' Get an itk image given an image filename of extionsion *TIFF, JPEG,
    PNG, BMP, DICOM, GIPL, Bio-Rad, LSM, Nifti, Analyze, SDT/SPR (Stimulate),
    Nrrd or VTK images*.'''

    reader = itk.ImageFileReader()
    reader.SetFileName(filename)

    image = reader.Execute()

    return image

def get_itk_array(filenameOrImage,normalize=False):
    ''' Get an image array given an image filename of extension *TIFF, JPEG,
    PNG, BMP, DICOM, GIPL, Bio-Rad, LSM, Nifti, Analyze, SDT/SPR (Stimulate),
    Nrrd or VTK images*.'''

    if isinstance(filenameOrImage,str):
        image = get_itk_image(filenameOrImage)
    else:
        image = filenameOrImage

    imageArray = itk.GetArrayFromImage(image)
    if normalize:
        imageArray = imageArray - np.min(imageArray)
        imageArray = imageArray*1.0 / np.max(imageArray)

    return imageArray

def get_itk_data(filenameOrImage, verbose=False):
    ''' Get the image array, image size and pixel dimensions of a certain itk
    image or an image specified by a certain filename (filenameOrImage).'''

    if isinstance(filenameOrImage,str):
        image = get_itk_image(filenameOrImage)
    else:
        image = filenameOrImage

    imageData = itk.GetArrayFromImage(image)
    imageSize = imageData.shape
    imageSpaces = image.GetSpacing()[::-1]
    imageDataType = imageData.dtype

    if verbose:
        print('\t image size: '+str(imageSize))
        print('\t image spacing: '+str(imageSpaces))
        print('\t image data type: '+str(imageDataType))

    return imageData, imageSize, imageSpaces

def convert_to_nii(filenames):
    ''' Convert image files to nifti image files.'''

    for filename in filenames:
        image = get_itk_image(filename)
        nii_filename = os.path.splitext(filename)[0]  + '.nii'
        write_itk_image(image,nii_filename)

    return

def convert_dicom(source_path, save_path):
    '''
    converts dicom series to image format specified in path.

    Parameters
    ----------
    source_path : string
        path to dicom series.
    path : string
        path to save new image (extension determines image format)

    '''

    image = read_dicom(source_path, verbose=True)

    write_itk_image(image, save_path)

def read_dicom(source_path, verbose=True):
    '''
    reads dicom series.

    Parameters
    ----------
    source_path : string
        path to dicom series.
    verbose : boolean
        print out all series file names.

    Returns
    -------
    image : itk image
        image volume.
    '''

    reader = itk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(source_path)
    if len(names) < 1:
        raise IOError('No Series can be found at the specified path!')
    elif verbose:
        print('image series found in :\n\t %s' % source_path)
    reader.SetFileNames(names)
    image = reader.Execute()
    if verbose:
        get_itk_data(image,verbose=True)

    return image

def read_images(img_index=0):
    patients = os.listdir(DATA_PATH)
    patients.sort()
    print("Found %i patients in the data path"%len(patients))
    data = []
    for p in patients:
        img = get_itk_image(DATA_PATH+p+"/"+names_dict[img_index])
        data.append(img)
    return data

def get_2d_images(inputfn,outfolder,ext):
    vol = get_itk_array(inputfn)
    import matplotlib.pyplot as ml
    for index ,img in enumerate(vol):
        outfn = outfolder + '/'+ str(index) + '.'+ext
        ml._imsave(fname=outfn,arr=img,cmap='gray')

def get_patch_data(volume4d, divs = (2,2,2,1), offset=(5,5,5,0)):
    patches = []
    shape = volume4d.shape
    widths = [ int(s/d) for s,d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths,offset)]
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                for t in np.arange(0, shape[3], widths[3]):
                    patch = np.zeros(patch_shape, dtype=volume4d.dtype)
                    x_s = x - offset[0] if x - offset[0] >= 0 else 0
                    x_e = x + widths[0] + offset[0] if x + widths[0] + offset[0] <= shape[0] else shape[0]
                    y_s = y - offset[1] if y - offset[1] >= 0 else 0
                    y_e = y + widths[1] + offset[1] if y + widths[1] + offset[1] <= shape[1] else shape[1]
                    z_s = z - offset[2] if z - offset[2] >= 0 else 0
                    z_e = z + widths[2] + offset[2] if z + widths[2] + offset[2] <= shape[2] else shape[2]
                    t_s = t - offset[3] if t - offset[3] >= 0 else 0
                    t_e = t + widths[3] +  offset[3] if t + widths[3] + offset[3] <= shape[3] else shape[3]
                    vp = volume4d[x_s:x_e,y_s:y_e,z_s:z_e,t_s:t_e]
                    px_s = offset[0] - (x - x_s)
                    px_e = px_s + (x_e - x_s)
                    py_s = offset[1] - (y - y_s)
                    py_e = py_s + (y_e - y_s)
                    pz_s = offset[2] - (z - z_s)
                    pz_e = pz_s + (z_e - z_s)
                    pt_s = offset[3] - (t - t_s)
                    pt_e = pt_s + (t_e - t_s)
                    patch[px_s:px_e,py_s:py_e,pz_s:pz_e,pt_s:pt_e] = vp
                    patches.append(patch)
    return np.array(patches, dtype=volume4d.dtype)

def get_volume_from_patches(patches5d, divs = (2,2,2,1), offset=(5,5,5,0)):
    new_shape = [(ps-of*2)*d for ps,of,d in zip(patches5d.shape[-4:],offset,divs)]
    volume4d = np.zeros(new_shape, dtype=patches5d.dtype)
    shape = volume4d.shape
    widths = [ int(s/d) for s,d in zip(shape,divs)]
    index = 0
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                for t in np.arange(0, shape[3], widths[3]):
                    patch = patches5d[index]
                    index = index + 1
                    volume4d[x:x+widths[0],y:y+widths[1],z:z+widths[2],t:t+widths[3]] = patch[offset[0]:offset[0]+widths[0],offset[1]:offset[1]+widths[1],offset[2]:offset[2]+widths[2],offset[3]:offset[3]+widths[3]]
    return volume4d
