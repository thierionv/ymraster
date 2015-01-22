# -*- coding: utf-8 -*-

"""
Raster manipulation library
===========================

This module contains classes for manipulating raster images. It is based on:

    * GDAL_ for general raster reading,

    * NumPy_ for computations,

    * rasterio_ for reading and writing raster efficiently

    * OTB_ for merge, concatenate and segmentation operations

:: _GDAL: http://gdal.org/
:: _NumPy: http://www.numpy.org/
:: _rasterio: https://github.com/mapbox/rasterio
:: _OTB: http://www.orfeo-toolbox.org/CookBook/


The ``Raster`` class
--------------------

The ``Raster`` class define an Image readed from a file.
"""

# from osgeo import gdal
import numpy as np
import rasterio

# TODO: Gérer plus tard les chemins d'accès qui ne sont pas les mêmes pour
# toutes les machines
import os
import sys
sys.path.append('/usr/lib/otb/python')
os.environ["ITK_AUTOLOAD_PATH"] = "/usr/lib/otb/applications"
import otbApplication


def _save_array(array, out_filename, meta):
    """Write an NumPy array to an image file

    :param array: the NumPy array to save
    :param out_filename: path to the file to write in
    :param meta: dict about the image (height, size, data type (int16,
    float64, etc.), projection, ...)
    """
    if array.ndim >= 4:
        raise NotImplementedError('Do not support 4+-dimensional arrays')
    with rasterio.drivers():
        with rasterio.open(out_filename, 'w', **meta) as raster:
            number_bands = meta['count']
            if number_bands == 1:
                raster.write_band(1, array)
            else:
                for i in range(number_bands):
                    raster.write_band(i+1, array[:, :, i])


class Raster():
    """Represents a raster image that was read from a file

    The whole raster *is not* loaded into memory. Instead this class records
    useful information about the raster (number and position of bands,
    resolution, ...) and provide useful methods for comparing rasters,
    computing some indices, etc.

    """

    def __init__(self, filename, *bands):
        """Create a new raster object read from a filename, and compute
        useful properties

        :param filename: a string containing the path of the image to read
        :param bands: band names (eg. 'blue', 'red', 'infrared, etc.)
        """
        self.filename = filename
        # Create 'idx_blue', 'idx_green', etc. attributes
        self.__dict__.update({'idx_'+band: i for i, band in enumerate(bands)})

        # Read information from image
        with rasterio.drivers():
            with rasterio.open(self.filename) as raster:
                self.meta = raster.meta

    def array(self):
        """Return a Numpy array corresponding to the image"""
        # Initialize an empty array of correct size and type
        array = np.empty((self.height,
                          self.width,
                          self.number_bands),
                         dtype='float64')

        # Fill the array
        with rasterio.drivers(CPL_DEBUG=True):  # Register GDAL format drivers
            with rasterio.open(self.filename) as img:
                for i in range(self.number_bands):
                    array[:, :, i] = img.read_band(i+1)
        return array

    def ndvi_array(self):
        """Return the Normlized Difference Vegetation Index (NDVI) of the image
        """
        array = self.array()
        band_red = array[:, :, self.idx_red]
        band_infrared = array[:, :, self.idx_infrared]
        band_red = np.where(band_infrared + band_red == 0, 1, band_red)
        return (band_infrared - band_red) / (band_infrared + band_red)

    def ndmi_array(self):
        """Return the Normalized Difference Moisture Index (NDMI) of the image
        """
        array = self.array()
        band_infrared = array[:, :, self.idx_infrared]
        band_midred = array[:, :, self.idx_midred]
        band_infrared = np.where(band_midred + band_infrared == 0, 1,
                                 band_infrared)
        return (band_infrared - band_midred) / (band_infrared + band_midred)

    def ndsi_array(self):
        """Return the Normalized Difference Snow Index (NDSI) of the image"""
        array = self.array()
        band_green = array[:, :, self.idx_green]
        band_midred = array[:, :, self.idx_midred]
        band_green = np.where(band_midred + band_green == 0, 1, band_green)
        return (band_green - band_midred) / (band_green + band_midred)

    def fusion(self, pan, output_image):
        """ Write the merge result between the two images of a bundle, using
        the BundleToPerfectSensor OTB application

        pan : a Raster instance of the panchromatic image
        output_image : path and name of the output image
        """
        Pansharpening = otbApplication.Registry.CreateApplication(
            "BundleToPerfectSensor")
        Pansharpening.SetParameterString("inp", pan.filename)
        Pansharpening.SetParameterString("inxs", self.filename)
        Pansharpening.SetParameterString("out", output_image)
        # Pansharpening.SetParameterOutputImagePixelType("out", 3)
        Pansharpening.ExecuteAndWriteOutput()

    def ndvi(self, output_image):
        """ Write a ndvi image, using the RadiometricIndices OTB application

        output_image : path and name of the output image
        """
        RadiometricIndices = otbApplication.Registry.CreateApplication(
            "RadiometricIndices")
        RadiometricIndices.SetParameterString("in", self.filename)
        RadiometricIndices.SetParameterInt("channels.red", self.idx_red)
        RadiometricIndices.SetParameterInt("channels.nir", self.idx_infrared)
        RadiometricIndices.SetParameterStringList("list", ["Vegetation:NDVI"])
        RadiometricIndices.SetParameterString("out", output_image)
        RadiometricIndices.ExecuteAndWriteOutput()

    def concatenate(self, list_im, output_image):
        """Concatenate a list of images of the same size into a single
        multi-band image,

        list_im : a list of raster instances
        output_image : path and name of the output image
        """
        # Creates a new list of the path image from the list of raster instances
        list_path = [self.filename]
        list_path.extend([im.filename for im in list_im])

        ConcatenateImages = otbApplication.Registry.CreateApplication(
            "ConcatenateImages")
        ConcatenateImages.SetParameterStringList("il", list_path)
        ConcatenateImages.SetParameterString("out", output_image)
        ConcatenateImages.ExecuteAndWriteOutput()

    def lsms_smoothing(self, output_filtered_image, spatialr, ranger, maxiter,
                       output_spatial_image=''):
        """First step of the segmentation: perform a mean shift fitlering,
        using the MeanShiftSmoothing otb application

        output_filtered_image : path and name of the output image filtered
        output_spatial_image : path and name of the output spatial image, the
        default value is an empty string, as this parameter is optional in
        MeanSiftSmoothing application
        spatialr : Int, Spatial radius of the neighborhooh
        ranger: Float, Range radius defining the radius (expressed in radiometry
        unit) in the multi-spectral space.
        maxiter : Int, Maximum number of iterations of the algorithm used in
            MeanSiftSmoothing application
        """

        # TODO : fix the paramaters and provide the posibility to the user to
        # set them

        MeanShiftSmoothing = otbApplication.Registry.CreateApplication(
            "MeanShiftSmoothing")
        MeanShiftSmoothing.SetParameterString("in", self.filename)
        MeanShiftSmoothing.SetParameterString("fout", output_filtered_image)
        MeanShiftSmoothing.SetParameterString("foutpos", output_spatial_image)
        MeanShiftSmoothing.SetParameterInt("spatialr", spatialr)
        MeanShiftSmoothing.SetParameterFloat("ranger", ranger)
        MeanShiftSmoothing.SetParameterFloat("thres", 0.1)
        MeanShiftSmoothing.SetParameterFloat("rangeramp", 0.1)
        MeanShiftSmoothing.SetParameterInt("maxiter", maxiter)
        MeanShiftSmoothing.ExecuteAndWriteOutput()

    def lsms_seg(self, input_pos_img, output_image, spatialr, ranger):
        """Second step of the segmentation: produce a labeled image with
        different clusters, using the LSMSSegmentation otb application

        input_pos_img : Raster instance of a spatial image, which may have been
        created in the smoothing step
        output_image : path and name of the output labeled image
        spatialr : Int, Spatial radius of the neighborhooh
        ranger: Float, Range radius defining the radius (expressed in radiometry
        unit) in the multi-spectral space.
        """
        LSMSSegmentation = otbApplication.Registry.CreateApplication(
            "LSMSSegmentation")
        LSMSSegmentation.SetParameterString("in", self.filename)
        LSMSSegmentation.SetParameterString("inpos", input_pos_img.filename)
        LSMSSegmentation.SetParameterString("out", output_image)
        LSMSSegmentation.SetParameterFloat("ranger", ranger)
        LSMSSegmentation.SetParameterFloat("spatialr", spatialr)
        LSMSSegmentation.SetParameterInt("minsize", 0)
        LSMSSegmentation.SetParameterInt("tilesizex", 256)
        LSMSSegmentation.SetParameterInt("tilesizey", 256)
        LSMSSegmentation.ExecuteAndWriteOutput()
