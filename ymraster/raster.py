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

from osgeo import osr, gdal
gdal.UseExceptions()
import numpy as np
import rasterio

# TODO: Gérer plus tard les chemins d'accès qui ne sont pas les mêmes pour
# toutes les machines
import os
from tempfile import gettempdir
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


def concatenate_images(rasters, out_filename):
    """Write an image which is the concatenation of the given rasters in order

    All bands in all input rasters must have same size

    Moreover, if data types are different, then everything will be converted to
    the default data type in OTB (_float_ currently).

    :param filenames: list of Raster objects to concatenate
    :param out_filename: path to the file to write the concatenation in
    """
    # Check for size and type (and that rasters list is not empty)
    raster0 = rasters[0]
    width, height = (raster0.meta['width'], raster0.meta['height'])
    otb_dtype = raster0.meta['dtype'].otb_dtype
    same_type = True
    for raster in rasters:
        assert raster.meta['width'] == width \
            and raster.meta['height'] == height, \
            "Images have not same size : '{}' and '{}'".format(raster0, raster)
        if raster.meta['dtype'].otb_dtype != otb_dtype:
            same_type = False

    # Perform the concatenation
    filenames = [raster.filename for raster in rasters]
    ConcatenateImages = otbApplication.Registry.CreateApplication(
        "ConcatenateImages")
    ConcatenateImages.SetParameterStringList("il", filenames)
    ConcatenateImages.SetParameterString("out", out_filename)
    if same_type:
        ConcatenateImages.SetParameterOutputImagePixelType("out", otb_dtype)
    ConcatenateImages.ExecuteAndWriteOutput()


class IdDefaultDict(dict):
    """A dictionary where trying to reach a missing key does create the key with
    value equal to itself"""

    def __missing__(self, key):
        self[key] = key
        return self[key]


class DataType(object):
    """Abstract class for a data type (int16, int32, float32, etc.)"""

    data_type_match = IdDefaultDict()

    def __set__(self, instance, value):
        instance.otb_dtype = self.data_type_match[value]

    def __get__(self, instance, owner):
        revert_match = {v: k for k, v in self.data_type_match.iteritems()}
        return revert_match[instance.otb_dtype]


class LStrDataType(DataType):
    """Represent a data type given in lower string format (eg. 'int16', 'int32',
    'float32', etc.)"""

    data_type_match = {'uint8': otbApplication.ImagePixelType_uint8,
                       'uint16': otbApplication.ImagePixelType_uint16,
                       'uint32': otbApplication.ImagePixelType_uint32,
                       'int16': otbApplication.ImagePixelType_int16,
                       'int32': otbApplication.ImagePixelType_int32,
                       'float32': otbApplication.ImagePixelType_float,
                       'float64': otbApplication.ImagePixelType_double}


class UStrDataType(DataType):
    """Represent a data type given in upper string format (eg. 'Int16', 'Int32',
    'Float32', etc.)"""

    data_type_match = {'UInt8': otbApplication.ImagePixelType_uint8,
                       'UInt16': otbApplication.ImagePixelType_uint16,
                       'UInt32': otbApplication.ImagePixelType_uint32,
                       'Int16': otbApplication.ImagePixelType_int16,
                       'Int32': otbApplication.ImagePixelType_int32,
                       'Float32': otbApplication.ImagePixelType_float,
                       'Float64': otbApplication.ImagePixelType_double}


class NumpyDataType(DataType):
    """Represent a data type for Numpy (eg. np.int16, np.int32, np.float32,
    etc.)"""

    data_type_match = {np.uint8: otbApplication.ImagePixelType_uint8,
                       np.uint16: otbApplication.ImagePixelType_uint16,
                       np.uint32: otbApplication.ImagePixelType_uint32,
                       np.int16: otbApplication.ImagePixelType_int16,
                       np.int32: otbApplication.ImagePixelType_int32,
                       np.float32: otbApplication.ImagePixelType_float,
                       np.float64: otbApplication.ImagePixelType_double}


class GdalDataType(DataType):
    """Represent a data type for gdal (eg. gdal.GDT_Int16, gdal.GDT_Iint32,
    gdal.GDT_Float32, etc.)"""

    data_type_match = {gdal.GDT_Byte: otbApplication.ImagePixelType_uint8,
                       gdal.GDT_UInt16: otbApplication.ImagePixelType_uint16,
                       gdal.GDT_UInt32: otbApplication.ImagePixelType_uint32,
                       gdal.GDT_Int16: otbApplication.ImagePixelType_int16,
                       gdal.GDT_Int32: otbApplication.ImagePixelType_int32,
                       gdal.GDT_Float32: otbApplication.ImagePixelType_float,
                       gdal.GDT_Float64: otbApplication.ImagePixelType_double}


class OtbDataType(DataType):
    """Represent a data type for orfeo-toolbox
    (eg. otbApplication.ImagePixelType_int16)"""

    def __set__(self, instance, value):
        instance._otb_type = value

    def __get__(self, instance, owner):
        return instance._otb_type


class RasterDataType(object):
    """The usable class to manage raster data types"""

    lstr_dtype = LStrDataType()
    ustr_dtype = UStrDataType()
    numpy_dtype = NumpyDataType()
    gdal_dtype = GdalDataType()
    otb_dtype = OtbDataType()

    def __init__(self,
                 lstr_dtype=None,
                 ustr_dtype=None,
                 numpy_dtype=None,
                 otb_dtype=None,
                 gdal_dtype=None):
        if lstr_dtype:
            self.lstr_dtype = lstr_dtype
        elif ustr_dtype:
            self.ustr_dtype = ustr_dtype
        elif numpy_dtype:
            self.numpy_dtype = numpy_dtype
        elif gdal_dtype:
            self.gdal_dtype = gdal_dtype
        elif otb_dtype:
            self.otb_dtype = otb_dtype


class Raster():
    """Represents a raster image that was read from a file

    The whole raster *is not* loaded into memory. Instead this class records
    useful information about the raster (number and position of bands,
    resolution, ...) and provide useful methods for comparing rasters,
    computing some indices, etc.

    """

    def __init__(self, filename):
        """Create a new raster object read from a filename, and compute
        useful properties

        :param filename: a string containing the path of the image to read
        :param bands: band names (eg. 'blue', 'red', 'infrared, etc.)
        """
        self.filename = filename

        # Read information from image
        ds = gdal.Open(self.filename, gdal.GA_ReadOnly)
        self.meta = {}
        self.meta['driver'] = ds.GetDriver()            # gdal.Driver object
        self.meta['count'] = ds.RasterCount             # int
        self.meta['width'] = ds.RasterXSize             # int
        self.meta['height'] = ds.RasterYSize            # int
        self.meta['dtype'] = RasterDataType(
            gdal_dtype=ds.GetRasterBand(1).DataType)    # RasterDataType object
        self.meta['transform'] = ds.GetGeoTransform()   # tuple

        # Read spatial reference as a osr.SpatialReference object or None
        # if there is no srs in metadata
        self.meta['srs'] = osr.SpatialReference(ds.GetProjection()) \
            if ds.GetProjection() \
            else None

        # Close file
        ds = None

    def array(self):
        """Return a Numpy array corresponding to the image"""
        # Initialize an empty array of correct size and type
        array = np.empty((self.meta['height'],
                          self.meta['width'],
                          self.meta['count']),
                         dtype=self.meta['dtype'].numpy_dtype)

        # Fill the array
        with rasterio.drivers(CPL_DEBUG=True):  # Register GDAL format drivers
            with rasterio.open(self.filename) as img:
                for i in range(self.meta['count']):
                    array[:, :, i] = img.read_band(i+1)
        return array

    def set_projection(self, srs):
        """Write the given projection into to file metadata

        :param srs: osgeo.osr.SpatialReference object that represents the
        projection to set
        """
        ds = gdal.Open(self.filename, gdal.GA_Update)
        ds.SetProjection(srs.ExportTowkt())
        ds = None
        self.meta['srs'] = srs

    def remove_band(self, idx, out_filename):
        """Write a new image with the band at the given index removed

        :param idx: index of the band to remove (starts at 1)
        :param out_filename: path to the output file
        """
        # Split the N-bands image into N mono-band images (in temp folder)
        SplitImage = otbApplication.Registry.CreateApplication("SplitImage")
        SplitImage.SetParameterString("in", self.filename)
        SplitImage.SetParameterString("out", os.path.join(gettempdir(),
                                                          'splitted.tif'))
        SplitImage.SetParameterOutputImagePixelType(
            "out",
            self.meta['dtype'].otb_dtype)
        SplitImage.ExecuteAndWriteOutput()

        # Concatenate the mono-band images without the unwanted band
        list_path = [os.path.join(gettempdir(), 'splitted_{}.tif'.format(i))
                     for i in range(self.meta['count'])
                     if i + 1 != idx]
        ConcatenateImages = otbApplication.Registry.CreateApplication(
            "ConcatenateImages")
        ConcatenateImages.SetParameterStringList("il", list_path)
        ConcatenateImages.SetParameterString("out", out_filename)
        ConcatenateImages.SetParameterOutputImagePixelType(
            "out",
            self.meta['dtype'].otb_dtype)
        ConcatenateImages.ExecuteAndWriteOutput()

        # Delete mono-band images in temp folder
        for i in range(self.meta['count']):
            os.remove(os.path.join(gettempdir(), 'splitted_{}.tif'.format(i)))

        return Raster(out_filename)

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

        return Raster(output_image)

    def ndvi(self, out_filename, idx_red, idx_nir):
        """Write the NDVI of the image into the given output file and
        return the corresponding Raster object. Indexation starts at 1.

        :param out_filename: path to the output file
        :param idx_red: index of the red band
        :param idx_nir: index of the near infrared band
        """
        RadiometricIndices = otbApplication.Registry.CreateApplication(
            "RadiometricIndices")
        RadiometricIndices.SetParameterString("in", self.filename)
        RadiometricIndices.SetParameterInt("channels.red", idx_red)
        RadiometricIndices.SetParameterInt("channels.nir", idx_nir)
        RadiometricIndices.SetParameterStringList("list", ["Vegetation:NDVI"])
        RadiometricIndices.SetParameterString("out", out_filename)
        RadiometricIndices.ExecuteAndWriteOutput()

        return Raster(out_filename)

    def ndwi(self, out_filename, idx_nir, idx_mir):
        """Write the NDWI of the image into the given output file and
        return the corresponding Raster object. Indexation starts at 1.

        :param out_filename: path to the output file
        :param idx_nir: index of the near infrared band
        :param idx_mir: index of the middle infrared band
        """
        RadiometricIndices = otbApplication.Registry.CreateApplication(
            "RadiometricIndices")
        RadiometricIndices.SetParameterString("in", self.filename)
        RadiometricIndices.SetParameterInt("channels.nir", idx_nir)
        RadiometricIndices.SetParameterInt("channels.mir", idx_mir)
        RadiometricIndices.SetParameterStringList("list", ["Water:NDWI"])
        RadiometricIndices.SetParameterString("out", out_filename)
        RadiometricIndices.ExecuteAndWriteOutput()

        return Raster(out_filename)

    ndmi = ndwi

    def mndwi(self, out_filename, idx_green, idx_mir):
        """Write the MNDWI of the image into the given output file and
        return the corresponding Raster object. Indexation starts at 1.

        :param out_filename: path to the output file
        :param idx_green: index of the green band
        :param idx_mir: index of the middle infrared band
        """
        RadiometricIndices = otbApplication.Registry.CreateApplication(
            "RadiometricIndices")
        RadiometricIndices.SetParameterString("in", self.filename)
        RadiometricIndices.SetParameterInt("channels.green", idx_green)
        RadiometricIndices.SetParameterInt("channels.mir", idx_mir)
        RadiometricIndices.SetParameterStringList("list", ["Water:MNDWI"])
        RadiometricIndices.SetParameterString("out", out_filename)
        RadiometricIndices.ExecuteAndWriteOutput()

        return Raster(out_filename)

    ndsi = mndwi

    def concatenate(self, list_im, output_image):
        """Concatenate a list of images of the same size into a single
        multi-band image,

        list_im : a list of raster instances
        output_image : path and name of the output image
        """
        rasters = [self]
        rasters.extend(list_im)
        concatenate_images(rasters, output_image)

        return Raster(output_image)

    def lsms_smoothing(self, output_filtered_image, spatialr, ranger, maxiter,
                       thres, rangeramp, output_spatial_image ):
        """First step of LSMS : perform a mean shift fitlering, using
        the MeanShiftSmoothing otb application. It returns two raster instances
        corresponding to the filtered image and the spatial image

        :param output_filtered_image : path and name of the output image
        filtered to be written
        :param output_spatial_image : path and name of the output spatial image
        to be written
        :param spatialr : Int, Spatial radius of the neighborhooh
        :param ranger: Float, Range radius defining the radius (expressed in
        radiometry unit) in the multi-spectral space.
        :param maxiter : Int, Maximum number of iterations of the algorithm
        used in MeanSiftSmoothing application
        :param thres : Float, Mean shift vector threshold
        :param rangeramp : Float, Range radius coefficient: This coefficient
        makes dependent the ranger of the colorimetry of the filtered pixel :
        y = rangeramp*x+ranger.
        """

        MeanShiftSmoothing = otbApplication.Registry.CreateApplication(
            "MeanShiftSmoothing")
        MeanShiftSmoothing.SetParameterString("in", self.filename)
        MeanShiftSmoothing.SetParameterString("fout", output_filtered_image)
        MeanShiftSmoothing.SetParameterString("foutpos", output_spatial_image)
        MeanShiftSmoothing.SetParameterInt("spatialr", spatialr)
        MeanShiftSmoothing.SetParameterFloat("ranger", ranger)
        MeanShiftSmoothing.SetParameterFloat("thres", thres)
        MeanShiftSmoothing.SetParameterFloat("rangeramp", rangeramp)
        MeanShiftSmoothing.SetParameterInt("maxiter", maxiter)
        MeanShiftSmoothing.ExecuteAndWriteOutput()

        return Raster(output_filtered_image), Raster(output_spatial_image)

    def lsms_seg (self,input_pos_img, output_seg_image, spatialr, ranger):
        """Second step of LSMS : produce a labeled image with different clusters,
        according to the range and spatial proximity of the pixels, using the
        LSMSSegmentation otb application. It returns a raster instance of the
        segmented image.

        :param input_pos_img : Raster instance of a spatial image, which may
        have been created in the smoothing step
        :param output_seg_image : path and name of the output segmented image
        to be written
        :param spatialr : Int, Spatial radius of the neighborhooh
        :param ranger: Float, Range radius defining the radius (expressed in
        radiometry unit) in the multi-spectral space.
        """
        LSMSSegmentation = otbApplication.Registry.CreateApplication(
            "LSMSSegmentation")
        LSMSSegmentation.SetParameterString("in", self.filename)
        LSMSSegmentation.SetParameterString("inpos", input_pos_img.filename)
        LSMSSegmentation.SetParameterString("out", output_seg_image)
        LSMSSegmentation.SetParameterFloat("ranger", ranger)
        LSMSSegmentation.SetParameterFloat("spatialr", spatialr)
        LSMSSegmentation.SetParameterInt("minsize", 0)
        LSMSSegmentation.SetParameterInt("tilesizex", 256)
        LSMSSegmentation.SetParameterInt("tilesizey", 256)
        LSMSSegmentation.ExecuteAndWriteOutput()

        return Raster(output_seg_image)

    def lsms_merging(self, in_smooth, output_merged, minsize):
        """Third step LSMS :  merge regions whose size in pixels is lower
        than minsize parameter with the adjacent regions with the adjacent
        region with closest radiometry and acceptable size, using the
        LSMSSmallRegionsMerging otb application. It returns a Raster instance
        of the merged image.

        :param in_smooth : Raster instance of the smoothed image, resulting
        from the step 1
        :param output_merged : path and name of the output merged segmented
        image to be written
        :param minsize : Int, minimum size of a label
        """

        # The following line creates an instance of the LSMSSmallRegionsMerging
        # application
        LSMSSmallRegionsMerging = otbApplication.Registry.CreateApplication(
            "LSMSSmallRegionsMerging")

        # The following lines set all the application parameters:
        LSMSSmallRegionsMerging.SetParameterString("in", in_smooth.filename)
        LSMSSmallRegionsMerging.SetParameterString("inseg", self.filename)
        LSMSSmallRegionsMerging.SetParameterString("out", output_merged)
        LSMSSmallRegionsMerging.SetParameterInt("minsize", minsize)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizex", 256)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizey", 256)

        # The following line execute the application
        LSMSSmallRegionsMerging.ExecuteAndWriteOutput()

        return Raster(output_merged)

    def lsms_vectorisation(self, in_image, output_vector):
        """Final step of LSMS : convert a label image to a GIS vector file
        containing one polygon per segment, using the LSMSVectorization otb
        application.

        :param in_image : Raster instance of the image
        :param output_vector : path and name of the output vector file ( ex:
        "vector.shp") to be written
        """
        # The following line creates an instance of the LSMSVectorization
        #application
        LSMSVectorization = otbApplication.Registry.CreateApplication(
            "LSMSVectorization")

        # The following lines set all the application parameters:
        LSMSVectorization.SetParameterString("in", in_image.filename)
        LSMSVectorization.SetParameterString("inseg", self.filename)
        LSMSVectorization.SetParameterString("out", output_vector)
        LSMSVectorization.SetParameterInt("tilesizex", 256)
        LSMSVectorization.SetParameterInt("tilesizey", 256)

        # The following line execute the application
        LSMSVectorization.ExecuteAndWriteOutput()

    def lsms(self, spatialr, ranger, maxiter, thres, rangeramp,
             output_filtered_image, output_spatial_image, output_seg_image,
             output_merged, minsize, output_vector, m_step = True):
        """Perform a segmentation on Raster instance given. It proceeds in 4
        steps in row : smoothing, segmentation, merging of small region and
        vectorisation.

        :param output_filtered_image : path and name of the output image
        filtered to be written
        :param output_spatial_image : path and name of the output spatial image
        to be written
        :param spatialr : Int, Spatial radius of the neighborhooh
        :param ranger: Float, Range radius defining the radius (expressed in
        radiometry unit) in the multi-spectral space.
        :param maxiter : Int, Maximum number of iterations of the algorithm
        used in MeanSiftSmoothing application
        :param thres : Float, Mode convergence threshold #TOCOMPLETE
        :param rangeramp : Float, Range radius coefficient: This coefficient
        makes dependent the ranger of the colorimetry of the filtered pixel :
        y = rangeramp*x+ranger.
        :param output_seg_image : path and name of the output segmented image
        to be written
        :param output_merged : path and name of the output merged segmented
        image to be written
        :param minsize : Int, minimum size of a label
        :param output_vector : path and name of the output vector file ( ex:
        "vector.shp") to be written
        :param m_step : Boolean, indicates if the merging step has to be done
        """

        img_smoothed, img_pos = self.lsms_smoothing(output_filtered_image, spatialr, ranger, maxiter, thres, rangeramp, output_spatial_image)

        print "smoothing step has been realized succesfully"

        img_seg = img_smoothed.lsms_seg(img_pos, output_seg_image, spatialr,
                                        ranger)

        print "segmentation step has been realized succesfully"

        if m_step :
            img_merged = img_seg.lsms_merging(img_smoothed, output_merged,
                                              minsize)

            print "merging step has been realized succesfully"

            img_merged.lsms_vectorisation(self, output_vector)

            print "vectorisation step has been realized succesfully"

        else:

            img_seg.lsms_vectorisation(self, output_vector)

            print "vectorisation step has been realized succesfully"
