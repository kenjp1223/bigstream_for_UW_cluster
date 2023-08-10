import xarray as xr
import zarr
from zarr.n5 import N5ChunkWrapper 
from skimage.transform import downscale_local_mean
import numpy as np
import pandas as pd


# read images from 
def imread(fname):
    with open(fname, 'rb') as fh:
        data = fh.read()
    return tiff_decode(data)

# Create zarr array from images
# The rootpath will be where all the lightsheet images data from SmartSpim will be contained.
# Channel_name should match the folder name which stores images you want to process, (e.g. Ex_785_Em_785_stitched)
# The metadata.txt should be stored in the rootpath. metadata flag will be FALSE when you want to specify parameters.
def create_zarr_from_SmartSpim2(rootpath,
                                channel_name,
                                metadata = True,
                                downscale_factor = (2,2,2),
                                chunks = (50, 128, 128),
                                compressor = zarr.GZip(level=-1),
                                xy_res = False,
                                z_res = False):
    # load high res image
    imagepath = os.path.join(rootpath,channel_name)
    image_highres_data = tifffile.imread([os.path.join(imagepath,f) for f in np.sort(os.listdir(imagepath)) if '.tif' in f],\
                                         aszarr=False, imread=imread)
    
    # Downsample the image
    image_lowres_data = downscale_local_mean(image_highres_data, downscale_factor)
    image_lowres_data = image_lowres_data.astype(image_highres_data.dtype)

    # Convert to xarray DataArrays
    highres_da = xr.DataArray(image_highres_data, dims=['z', 'y', 'x'], name='highres')
    lowres_da = xr.DataArray(image_lowres_data, dims=['z', 'y', 'x'], name='lowres')

    # Create a new Zarr store
    store_path = os.path.join(rootpath,channel_name + '.zarr')
    store = zarr.open_group(store_path, mode='w')
    
    # load metadata, if metadata flag FALSE use pre-set parameters
    if metadata:
        metapath = os.path.join(rootpath,'metadata.txt') 
        metadf = pd.read_csv(metapath, sep='\t', header=0,nrows = 1,encoding= 'unicode_escape')
        #print(metadf.columns)
        xy_res,z_res = metadf.loc[:,metadf.columns[2:4]].values[0]

    # Save the DataArrays as arrays within the Zarr store
    store.create_dataset('highres', data=image_highres_data, compressor=compressor, chunks=chunks)
    store.create_dataset('lowres', data=image_lowres_data, compressor=compressor, chunks=chunks)

    # Add attributes
    store['/highres'].attrs['pixelResolution'] = (z_res, xy_res, xy_res)
    store['/highres'].attrs['downsamplingFactor'] = (1, 1, 1)  # No downsampling for highres
    store['/lowres'].attrs['pixelResolution'] = (z_res, xy_res, xy_res)
    store['/lowres'].attrs['downsamplingFactor'] = downscale_factor  
