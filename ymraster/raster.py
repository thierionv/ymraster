# -*- coding: utf-8 -*-

"""
A ``Raster`` instance represents a raster read from a file.

>>> raster = Raster('tests/data/RGB.byte.tif')

It has some attributes:

>>> raster.filename
'tests/data/RGB.byte.tif'
>>> raster.meta['width']
791

Functions and methods
=====================
"""

try:
    import otbApplication as otb
except ImportError as e:
    raise ImportError(
        str(e)
        + "\n\nPlease install Orfeo Toolbox if it isn't installed yet.\n\n"
        "Also, add the otbApplication module path "
        "(usually something like '/usr/lib/otb/python') "
        "to the PYTHONPATH environment variable.")
try:
    app = otb.Registry.CreateApplication('Smoothing')
    app.SetParameterString('out', 'foo.tif')
except AttributeError:
    raise ImportError(
        "Unable to create otbApplication objects\n\n"
        "Please set the ITK_AUTOLOAD_PATH environment variable "
        "to the Orfeo Toolbox applications folder path "
        "(usually something like '/usr/lib/otb/applications') ")
try:
    from osgeo import osr, gdal
    gdal.UseExceptions()
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install GDAL.")
try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install NumPy.")
import dtype

from fix_proj_decorator import fix_missing_proj

from datetime import datetime
import os
from tempfile import gettempdir


def _save_array(array, out_filename, driver_name, dtype, proj=None,
                geotransform=None, date=None):
    """Write an NumPy array to an image file.

    :param array: the NumPy array to save
    :param out_filename: path to the file to write in
    :param meta: dict about the image (height, size, data type (int16,
    float64, etc.), projection, ...)
    """
    # Get array size
    if array.ndim >= 4:
        raise NotImplementedError('Do not support 4+-dimensional arrays')
    if array.ndim == 3:
        ysize, xsize, number_bands = array.shape
    else:
        ysize, xsize = array.shape
        number_bands = 1

    # Create an empty raster of correct size
    driver = gdal.GetDriverByName(driver_name)
    out_raster = driver.Create(out_filename,
                               xsize,
                               ysize,
                               number_bands,
                               dtype.gdal_dtype)
    if proj is not None:
        out_raster.SetProjection(proj)
    if geotransform is not None:
        out_raster.SetGeoTransform(geotransform)
    if number_bands == 1:
        band = out_raster.GetRasterBand(1)
        band.WriteArray(array)
        band.FlushCache()
    else:
        for i in range(number_bands):
            band = out_raster.GetRasterBand(i+1)
            band.WriteArray(array[:, :, i])
            band.FlushCache()
    out_raster = None
    band = None


def concatenate_images(rasters, out_filename):
    """Write a raster which is the concatenation of the given rasters, in order.

    All bands in all input rasters must have same size.

    Moreover, this function is quite conservative about projections: all bands
    should have the same projection

    Finally, if data types are different, then everything will be converted to
    the default data type in OTB (_float_ currently).

    :param rasters: the ``Raster`` instances to concatenate
    :type rasters: list of ``Raster`` instances
    :param out_filename: path to the output file
    :type out_filename: str to the output file
    """
    # Check for size, proj, transform & type (and that list not empty)
    raster0 = rasters[0]
    srs, otb_dtype = (raster0.meta['srs'], raster0.meta['dtype'].otb_dtype)
    assert srs is not None, \
        "Image has no Coordinate Reference System : '{}'".format(
            raster0.filename)
    same_type = True
    for raster in rasters:
        assert raster.meta['srs'] is not None \
            and raster.meta['srs'].IsSame(srs), \
            "Images have not the same Coordinate Reference System : "
        "'{}' and '{}'".format(raster0.filename, raster.filename)
        if raster.meta['dtype'].otb_dtype != otb_dtype:
            same_type = False

    # Perform the concatenation
    filenames = [raster.filename for raster in rasters]
    ConcatenateImages = otb.Registry.CreateApplication("ConcatenateImages")
    ConcatenateImages.SetParameterStringList("il", filenames)
    ConcatenateImages.SetParameterString("out", out_filename)
    if same_type:
        ConcatenateImages.SetParameterOutputImagePixelType("out", otb_dtype)
    ConcatenateImages.ExecuteAndWriteOutput()


class Raster():
    """Represents a raster image that was read from a file.

    The whole raster *is not* loaded into memory. Instead this class records
    useful information about the raster (number and size of bands, projection,
    etc.) and provide useful methods for comparing rasters, computing some
    indices, etc.

    """

    def __init__(self, filename):
        """Create a new raster object read from a file, and compute useful
        properties.

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
        self.meta['dtype'] = dtype.RasterDataType(      # RasterDataType object
            gdal_dtype=ds.GetRasterBand(1).DataType)
        self.meta['transform'] = ds.GetGeoTransform(    # tuple
            can_return_null=True)
        try:
            self.meta['datetime'] = datetime.strptime(  # datetime object
                ds.GetMetadataItem('TIFFTAG_DATETIME'), '%Y:%m:%d %H:%M:%S')
        except ValueError:  # string has wrong datetime format
            self.meta['datetime'] = None
        except TypeError:   # there is no DATETIME tag
            self.meta['datetime'] = None

        # Read spatial reference as a osr.SpatialReference object or None
        # if there is no srs in metadata
        self.meta['srs'] = osr.SpatialReference(ds.GetProjection()) \
            if ds.GetProjection() \
            else None

        # Close file
        ds = None

    def array(self):
        """Returns the NumPy array corresponding to the raster.

        :rtype: numpy.ndarray"""
        # Initialize an empty array of correct size and type
        array = np.empty((self.meta['height'],
                          self.meta['width'],
                          self.meta['count']),
                         dtype=self.meta['dtype'].numpy_dtype)

        # Fill the array
        ds = gdal.Open(self.filename, gdal.GA_ReadOnly)
        for i in range(self.meta['count']):
            array[:, :, i] = ds.GetRasterBand(i+1).ReadAsArray()
        ds = None

        return array

    def set_projection(self, srs):
        """Writes the given projection into the raster's metadata.

        :param srs: projection to set
        :type srs: osgeo.osr.SpatialReference
        """
        ds = gdal.Open(self.filename, gdal.GA_Update)
        ds.SetProjection(srs.ExportToWkt())
        ds = None
        self.meta['srs'] = srs

    def set_datetime(self, dt):
        """Writes the given datetime into the raster's metadata.

        :param dt: datetime to set
        :type dt: datetime.datetime
        """
        ds = gdal.Open(self.filename, gdal.GA_Update)
        ds.SetMetadata({'TIFFTAG_DATETIME': dt.strftime('%Y:%m:%d %H:%M:%S')})
        ds = None
        self.meta['datetime'] = dt

    def remove_band(self, idx, out_filename):
        """Writes a new raster (in the specified output file) which is the same
        than the current raster, except that the band at the given index has
        been remove.

        :param idx: index of the band to remove (starts at 1)
        :type idx: int
        :param out_filename: path to the output file
        :type out_filename: str
        :returns: the ``Raster`` instance corresponding to the output file
        """
        # Split the N-bands image into N mono-band images (in temp folder)
        SplitImage = otb.Registry.CreateApplication("SplitImage")
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
        ConcatenateImages = otb.Registry.CreateApplication("ConcatenateImages")
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

    def fusion(self, pan, out_filename):
        """Sharpen the raster with a more detailed panchromatic image, and save
        the result into the specified output file.

        :param pan: panchromatic image to use for sharpening
        :type pan: ``Raster``
        :param out_filename: path to the output file
        :type out_filename: str
        :returns: the ``Raster`` instance corresponding to the output file
        """
        assert (self.meta['srs'] is not None
                and pan.meta['srs'] is not None
                and self.meta['srs'].IsSame(pan.meta['srs'])) \
            or (self.meta['srs'] is None
                and pan.meta['srs'] is None), \
            "Images have not the same Coordinate Reference System : "
        "'{}' and '{}'".format(self.filename, pan.filename)
        Pansharpening = otb.Registry.CreateApplication("BundleToPerfectSensor")
        Pansharpening.SetParameterString("inp", pan.filename)
        Pansharpening.SetParameterString("inxs", self.filename)
        Pansharpening.SetParameterString("out", out_filename)
        # Pansharpening.SetParameterOutputImagePixelType("out", 3)
        Pansharpening.ExecuteAndWriteOutput()

        return Raster(out_filename)

    @fix_missing_proj
    def ndvi(self, out_filename, idx_red, idx_nir):
        """Writes the Normalized Difference Vegetation Index (NDVI) of the
        raster into the specified output file.

        :param out_filename: path to the output file
        :type out_filename: str
        :param idx_red: index of a red band (starts at 1)
        :type idx_red: int
        :param idx_nir: index of a near-infrared band (starts at 1)
        :type idx_nir: int
        :returns: the ``Raster`` instance corresponding to the output file
        """
        RadiometricIndices = otb.Registry.CreateApplication(
            "RadiometricIndices")
        RadiometricIndices.SetParameterString("in", self.filename)
        RadiometricIndices.SetParameterInt("channels.red", idx_red)
        RadiometricIndices.SetParameterInt("channels.nir", idx_nir)
        RadiometricIndices.SetParameterStringList("list", ["Vegetation:NDVI"])
        RadiometricIndices.SetParameterString("out", out_filename)
        RadiometricIndices.ExecuteAndWriteOutput()

        return Raster(out_filename)

    @fix_missing_proj
    def ndwi(self, out_filename, idx_nir, idx_mir):
        """Writes the Normalized Difference Vegetation Index (NDWI) of the
        raster into the given output file.

        :param out_filename: path to the output file
        :type out_filename: str
        :param idx_nir: index of the near infrared band (starts at 1)
        :type idx_nir: int
        :param idx_mir: index of the middle infrared band (starts at 1)
        :type idx_mir: int
        :returns: the ``Raster`` instance corresponding to the output file
        """
        RadiometricIndices = otb.Registry.CreateApplication(
            "RadiometricIndices")
        RadiometricIndices.SetParameterString("in", self.filename)
        RadiometricIndices.SetParameterInt("channels.nir", idx_nir)
        RadiometricIndices.SetParameterInt("channels.mir", idx_mir)
        RadiometricIndices.SetParameterStringList("list", ["Water:NDWI"])
        RadiometricIndices.SetParameterString("out", out_filename)
        RadiometricIndices.ExecuteAndWriteOutput()

        return Raster(out_filename)

    ndmi = ndwi

    @fix_missing_proj
    def mndwi(self, out_filename, idx_green, idx_mir):
        """Writes the Modified Normalized Difference Water Index (MNDWI) of the
        image into the given output file.

        :param out_filename: path to the output file
        :type out_filename: str
        :param idx_green: index of the green band
        :type idx_green: int
        :param idx_mir: index of the middle infrared band
        :type idx_mir: int
        :returns: the ``Raster`` instance corresponding to the output file
        """
        RadiometricIndices = otb.Registry.CreateApplication(
            "RadiometricIndices")
        RadiometricIndices.SetParameterString("in", self.filename)
        RadiometricIndices.SetParameterInt("channels.green", idx_green)
        RadiometricIndices.SetParameterInt("channels.mir", idx_mir)
        RadiometricIndices.SetParameterStringList("list", ["Water:MNDWI"])
        RadiometricIndices.SetParameterString("out", out_filename)
        RadiometricIndices.ExecuteAndWriteOutput()

        return Raster(out_filename)

    ndsi = mndwi

    def concatenate(self, rasters, out_filename):
        """Concatenates into the specifed output file:
            * the current raster, *and*,
            * a list of rasters of the same size.

        :param rasters: a list of rasters to append after the current raster
        :type rasters: list of ``Raster``
        :param out_filename: path to the output file
        :type out_filename: str
        :returns: the ``Raster`` instance corresponding to the output file
        """
        list_ = [self] + rasters
        concatenate_images(list_, out_filename)
        return Raster(out_filename)

    def lsms_smoothing(self, out_smoothed_filename, spatialr, ranger,
                       out_spatial_filename, thres=0.1, rangeramp=0,
                       maxiter=10, modesearch=0):
        """First step of a Large-Scale Mean-Shift (LSMS) segmentation: performs
        a mean shift smoothing on the raster.

        This is an adapted version of the Orfeo Toolbox ``MeanShiftSmoothing``
        application. See
        http://www.orfeo-toolbox.org/CookBook/CookBooksu91.html#x122-5480005.5.2
        for more details

        :param out_smoothed_filename: path to the smoothed file to be written
        :type out_smoothed_filename: str
        :param out_spatial_filename: path to the spatial image to be written
        :type out_spatial_filename: str
        :param spatialr: spatial radius of the window
        :type spatialr: int
        :param ranger: range radius defining the spectral window size (expressed
                       in radiometry unit)
        :type ranger: float
        :param maxiter: maximum number of iterations in case of non-convergence
        :type maxiter: int
        :param thres: mean shift vector threshold
        :type thres: float
        :param rangeramp: range radius coefficient. This coefficient makes
                          dependent the ``ranger`` of the colorimetry of the
                          filtered pixel:
                          .. math::

                              y = rangeramp * x + ranger
        :type rangeramp: float
        :returns: two ``Raster`` instances corresponding to the filtered image
                  and the spatial image
        :rtype: tuple of ``Raster``
        """
        MeanShiftSmoothing = otb.Registry.CreateApplication(
            "MeanShiftSmoothing")
        MeanShiftSmoothing.SetParameterString("in", self.filename)
        MeanShiftSmoothing.SetParameterString("fout", out_smoothed_filename)
        MeanShiftSmoothing.SetParameterString("foutpos", out_spatial_filename)
        MeanShiftSmoothing.SetParameterInt("spatialr", spatialr)
        MeanShiftSmoothing.SetParameterFloat("ranger", ranger)
        MeanShiftSmoothing.SetParameterFloat("thres", thres)
        MeanShiftSmoothing.SetParameterFloat("rangeramp", rangeramp)
        MeanShiftSmoothing.SetParameterInt("maxiter", maxiter)
        MeanShiftSmoothing.SetParameterInt("modesearch", modesearch)
        MeanShiftSmoothing.ExecuteAndWriteOutput()

        return Raster(out_smoothed_filename), Raster(out_spatial_filename)

    def lsms_segmentation(self, in_spatial_raster, out_filename, spatialr,
                          ranger, tilesizex=256, tilesizey=256):
        """Second step in a LSMS segmentation: performs the actual object
        segmentation on the raster. Produce an image whose pixels are given a
        label number, based on their spectral and spatial proximity.

        This assumes that the ``Raster`` object is smoothed, for example as
        returned by the ``lsms_smoothing`` method.

        To consume less memory resources, the method tiles the raster and
        performs the segmentation on each tile.

        This is an adapted version of the Orfeo Toolbox ``LSMSSegmentation``
        application where there is no option to set a ``minsize`` parameter to
        discard small objects. See
        http://www.orfeo-toolbox.org/CookBook/CookBooksu121.html#x156-8990005.9.3
        for more details

        :param in_spatial_raster: a spatial raster associated to this raster
                                  (for example, as returned by the
                                  ``lsms_smoothing`` method)
        :type in_spatial_raster: ``Raster``
        :param out_filename: path to the segmented image to be written
        :param spatialr: spatial radius of the window
        :type spatialr: int
        :param ranger: range radius defining the spectral window size (expressed
                       in radiometry unit)
        :type ranger: float
        :param tilesizex: width of each tile (default: 256)
        :type tilesizex: int
        :param tilesizey: height of each tile (default: 256)
        :type tilesizey: int
        :returns: ``Raster`` instance corresponding to the segmented image
        """
        LSMSSegmentation = otb.Registry.CreateApplication("LSMSSegmentation")
        LSMSSegmentation.SetParameterString("in", self.filename)
        LSMSSegmentation.SetParameterString("inpos",
                                            in_spatial_raster.filename)
        LSMSSegmentation.SetParameterString("out", out_filename)
        LSMSSegmentation.SetParameterFloat("ranger", ranger)
        LSMSSegmentation.SetParameterFloat("spatialr", spatialr)
        LSMSSegmentation.SetParameterInt("minsize", 0)
        LSMSSegmentation.SetParameterInt("tilesizex", tilesizex)
        LSMSSegmentation.SetParameterInt("tilesizey", tilesizey)
        LSMSSegmentation.ExecuteAndWriteOutput()

        return Raster(out_filename)

    @fix_missing_proj
    def lsms_merging(self, in_smoothed_raster, out_filename, minsize,
                     tilesizex=256, tilesizey=256):
        """Optional third step in a LSMS segmentation:  merge objects in the
        raster whose size in pixels is lower than a given threshold into the
        bigger enough adjacent object with closest radiometry (radiometry is
        given by the original image from which the labeled raster was computed).

        This assumes that the ``Raster`` object is a segmented and labeled
        image, for example as returned by the ``lsms_segmentation`` method.

        To consume less memory resources, the method tiles the raster and
        performs the segmentation on each tile.

        This is an adapted version of the Orfeo Toolbox
        ``LSMSSmallRegionsMerging`` application. See
        http://www.orfeo-toolbox.org/CookBook/CookBooksu122.html#x157-9060005.9.4
        for more details.

        the LSMSSmallRegionsMerging otb application. It returns a Raster
        instance of the merged image.

        :param in_smoothed_raster: smoothed raster associated to this raster
                                   (for example, as returned by the
                                   ``lsms_smoothing`` method)
        :type in_smoothed_raster: ``Raster``
        :param out_filename: path to the merged segmented image to be written
        :type out_filename: str
        :param minsize: threshold defining the minimum size of an object
        :type minsize: int
        :param tilesizex: width of each tile (default: 256)
        :type tilesizex: int
        :param tilesizey: height of each tile (default: 256)
        :type tilesizey: int
        :returns: ``Raster`` instance corresponding to the merged segmented
                  image
        """
        LSMSSmallRegionsMerging = otb.Registry.CreateApplication(
            "LSMSSmallRegionsMerging")
        LSMSSmallRegionsMerging.SetParameterString("in",
                                                   in_smoothed_raster.filename)
        LSMSSmallRegionsMerging.SetParameterString("inseg", self.filename)
        LSMSSmallRegionsMerging.SetParameterString("out", out_filename)
        LSMSSmallRegionsMerging.SetParameterInt("minsize", minsize)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizex", tilesizex)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizey", tilesizey)
        LSMSSmallRegionsMerging.ExecuteAndWriteOutput()

        return Raster(out_filename)

    def lsms_vectorization(self, orig_raster, out_filename, tilesizex=256,
                           tilesizey=256):
        """Last step in a LSMS segmentation: vectorize a labeled segmented
        image, turn each object into a polygon. Each polygon will have some
        attribute data:

            * the label number as an attribute,
            * the object's mean for each band in the original image,
            * the object's standard deviation for each band in the original
              image,
            * number of pixels in the object.

        This assumes that the ``Raster`` object is a segmented and labeled
        image, for example as returned by the ``lsms_segmentation`` or the
        ``lsms_merging`` methods.

        To consume less memory resources, the method tiles the raster and
        performs the segmentation on each tile.

        to a vector file containing one polygon per segment, using the
        LSMSVectorization otb application.

        :param orig_raster: original raster from which the segmentation was
                             computed
        :param out_filename: path to the output vector file
        :param tilesizex: width of each tile (default: 256)
        :type tilesizex: int
        :param tilesizey: height of each tile (default: 256)
        :type tilesizey: int
        """
        LSMSVectorization = otb.Registry.CreateApplication(
            "LSMSVectorization")
        LSMSVectorization.SetParameterString("in", orig_raster.filename)
        LSMSVectorization.SetParameterString("inseg", self.filename)
        LSMSVectorization.SetParameterString("out", out_filename)
        LSMSVectorization.SetParameterInt("tilesizex", tilesizex)
        LSMSVectorization.SetParameterInt("tilesizey", tilesizey)
        LSMSVectorization.ExecuteAndWriteOutput()

    def lsms(self, spatialr, ranger, maxiter, thres, rangeramp,
             output_filtered_image, output_spatial_image, output_seg_image,
             output_merged, minsize, output_vector, m_step=True):
        img_smoothed, img_pos = self.lsms_smoothing(output_filtered_image,
                                                    spatialr,
                                                    ranger,
                                                    maxiter,
                                                    thres,
                                                    rangeramp,
                                                    output_spatial_image)

        print("smoothing step has been realized succesfully")

        img_seg = img_smoothed.lsms_seg(img_pos, output_seg_image, spatialr,
                                        ranger)

        print("segmentation step has been realized succesfully")

        if m_step:
            img_merged = img_seg.lsms_merging(img_smoothed, output_merged,
                                              minsize)

            print("merging step has been realized succesfully")

            img_merged.lsms_vectorisation(self, output_vector)

            print("vectorisation step has been realized succesfully")

        else:

            img_seg.lsms_vectorisation(self, output_vector)

            print("vectorisation step has been realized succesfully")
