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
import dtype as data_type
import array_stat

from fix_proj_decorator import fix_missing_proj

from datetime import datetime
from time import mktime
import os
from tempfile import gettempdir


def _dt2float(dt):
    """Returns a float corresponding to the given datetime object.

    :param dt: datetime to convert into a float
    :type dt: datetime.datetime
    :rtype: float
    """
    return mktime(dt.timetuple())


def write_file(out_filename, overwrite=False, drivername=None, dtype=None,
               array=None, width=None, height=None, depth=None, dt=None,
               srs=None, transform=None, xoffset=0, yoffset=0):
    """Writes a NumPy array to an image file.

    If there is no array (array is None), the function simply create an empty
    image file of given size (width, height, depth). In other words, if array is
    None, width, height and depth must be specified.

    If array is not None, then all other parameters are optional: if the file
    exists, array will be written into the file. If the file does not exists, a
    new file will be created with same size and type than the array.

    This allows to write a file block by block. First create an empty file of
    correct size and type. Then write each block into the file using the xoffset
    and yoffset parameters.

    :param out_filename: path to the output file
    :type out_filename: str
    :param overwrite: if True, overwrite file if exists. False by default.
    :type overwrite: bool
    :param drivername: name of the driver to use. None means that the file
                       already exists
    :type drivername: str
    :param dtype: datatype to use for the output file. None means that the file
                  already exists
    :type dtype: RasterDataType
    :param array: the NumPy array to save. None means that an empty file will be
                  created or, if file exists, that no data will be written
                  (except metadata if specified)
    :type array: np.ndarray
    :param width: horizontal size of the image to be created
    :type width: int
    :param height: vertical size of the image to be created
    :type width: int
    :param depth: number of bands of the image to be created
    :type depth: int
    :param dt: date/time to write in the output file metadata
    :type dt: datetime.datetime
    :param srs: projection to write in the output file metadata
    :type srs: osr.SpatialReference
    :param transform: geo-transformation to use for the output file
    :type transform: 6-tuple of floats
    :param xoffset: horizontal offset. First index from which to write the array
                    in the output file if the array is smaller (default: 0)
    :type xoffset: float
    :param yoffset: Vertical offset. First index from which to write the array
                    in the output file if the array is smaller (default: 0)
    :type yoffset: float

    """
    # Size & data type of output image
    xsize, ysize = (width, height) \
        if width and height \
        else (array.shape[1], array.shape[0])
    try:
        number_bands = depth if depth else array.shape[2]
    except IndexError:
        number_bands = 1
    datatype = dtype \
        if dtype \
        else data_type.RasterDataType(numpy_dtype=array.dtype)

    # Create an empty raster file if it does not exists or if overwrite is True
    try:
        assert not overwrite
        out_ds = gdal.Open(out_filename, gdal.GA_Update)
    except (AssertionError, RuntimeError):
        driver = gdal.GetDriverByName(drivername)
        out_ds = driver.Create(out_filename,
                               xsize,
                               ysize,
                               number_bands,
                               datatype.gdal_dtype)

    # Set metadata
    if dt:
        out_ds.SetMetadata(
            {'TIFFTAG_DATETIME': dt.strftime('%Y:%m:%d %H:%M:%S')})
    if srs:
        out_ds.SetProjection(srs.ExportToWkt())
    if transform:
        out_ds.SetGeoTransform(transform)

    # Save array if there is an array to save
    if array is None:
        return
    if number_bands == 1:
        band = out_ds.GetRasterBand(1)
        band.WriteArray(array, xoff=xoffset, yoff=yoffset)
        band.FlushCache()
    else:
        for i in range(number_bands):
            band = out_ds.GetRasterBand(i+1)
            band.WriteArray(array[:, :, i], xoff=xoffset, yoff=yoffset)
            band.FlushCache()
    band = None
    out_ds = None


def concatenate_images(rasters, out_filename):
    """Write a raster which is the concatenation of the given rasters, in order.

    All bands in all input rasters must have same size.

    Moreover, this function is quite conservative about projections: all bands
    should have the same projection and same extent

    Finally, if data types are different, then everything will be converted to
    the default data type in OTB (_float_ currently).

    :param rasters: the ``Raster`` instances to concatenate
    :type rasters: list of ``Raster`` instances
    :param out_filename: path to the output file
    :type out_filename: str to the output file
    """
    # Check for proj, extent & type (and that list not empty)
    raster0 = rasters[0]
    srs, extent, otb_dtype = (raster0.meta['srs'],
                              raster0.meta['gdal_extent'],
                              raster0.meta['dtype'].otb_dtype)
    assert srs is not None, \
        "Image has no Coordinate Reference System: '{}'".format(
            raster0.filename)
    same_type = True
    for raster in rasters:
        assert raster.meta['srs'] is not None \
            and raster.meta['srs'].IsSame(srs), \
            "Images have not the same Coordinate Reference System: "
        "'{}' and '{}'".format(raster0.filename, raster.filename)
        assert raster.meta['gdal_extent'] == extent, \
            "Images have not the same extent: "
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


def temporal_stats(rasters, out_filename, drivername, idx_band=1,
                   stats=['min', 'max'], date2float=_dt2float):
    """Compute pixel-wise statistics from a given list of temporally distinct,
    but spatially identical, rasters.

    Only one band in each raster is considered (by default: the first one).

    Output is a multi-band raster where each band contains a statistic (eg.
    max, mean). For summary statistics (eg. maximum), there is an additional
    band which gives the date/time at which the result has been found, in
    numeric format, as a result of the given date2float function (by default
    converts a date into seconds since 1970, eg. Apr 25, 2013 (midnight) ->
    1366840800.0)

    :param rasters: list of rasters to compute statistics from
    :type rasters: list of ``Raster`` instances
    :param out_filename: path to the output file
    :type out_filename: str
    :param drivername: driver to use for writing the output file
    :type drivername: str
    :param idx_band: index of the band to compute statistics on (default: 1)
    :type idx_band: int
    :param stats: list of stats to compute
    :type stats: list of str
    :param date2float: function which returns a float from a datetime object.
                       By default, it is the time.mktime() function
    :type date2float: function
    """
    # Number of bands in output file
    depth = len(stats) + len([stat for stat in stats
                              if array_stat.ArrayStat(stat).is_summary])

    # Create an empty file based on what is to be computed
    raster0 = rasters[0]
    write_file(out_filename,
               overwrite=True,
               drivername=drivername,
               dtype=data_type.RasterDataType(lstr_dtype='float64'),
               width=raster0.meta['width'],
               height=raster0.meta['height'],
               depth=depth,
               srs=raster0.meta['srs'],
               transform=raster0.meta['transform'])

    # TODO: improve to find better "natural" blocks than using the "natural"
    # segmentation of simply the first image
    block_wins = raster0.block_windows()

    for block_win in block_wins:
        # Turn each block into an array and concatenate them into a stack
        block_arrays = [raster.array(idx_band, block_win) for raster in rasters]
        block_stack = np.dstack(block_arrays) \
            if len(block_arrays) > 1 \
            else block_arrays[0]

        # Compute each stat for the block and append the result to a list
        stat_array_list = []
        for stat in stats:
            astat = array_stat.ArrayStat(stat, axis=2)
            stat_array_list.append(astat.compute(block_stack))
            if astat.is_summary:  # If summary stat, compute date of occurence
                date_array = astat.indices(block_stack)
                for x in np.nditer(date_array, op_flags=['readwrite']):
                    x[...] = date2float(rasters[x].meta['datetime'])
                stat_array_list.append(date_array)

        # Concatenate results into a stack and save the block to the output file
        stat_stack = np.dstack(stat_array_list) \
            if len(stat_array_list) > 1 \
            else stat_array_list[0]
        xoffset, yoffset = block_win[0], block_win[1]
        write_file(out_filename, array=stat_stack,
                   xoffset=xoffset, yoffset=yoffset)


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
        self.meta['width'] = ds.RasterXSize             # int
        self.meta['height'] = ds.RasterYSize            # int
        self.meta['count'] = ds.RasterCount             # int
        self.meta['dtype'] = data_type.RasterDataType(
            gdal_dtype=ds.GetRasterBand(1).DataType)    # RasterDataType object
        self.meta['nodata_value'] = \
            ds.GetRasterBand(1).GetNoDataValue()       # float
        self.meta['block_size'] = tuple(
            ds.GetRasterBand(1).GetBlockSize())         # tuple
        try:                                            # datetime object
            self.meta['datetime'] = datetime.strptime(
                ds.GetMetadataItem('TIFFTAG_DATETIME'), '%Y:%m:%d %H:%M:%S')
        except ValueError:  # string has wrong datetime format
            self.meta['datetime'] = None
        except TypeError:   # there is no DATETIME tag
            self.meta['datetime'] = None
        self.meta['transform'] = ds.GetGeoTransform(    # tuple
            can_return_null=True)
        self.meta['gdal_extent'] = tuple(               # tuple
            (ds.GetGeoTransform()[0]
             + x * ds.GetGeoTransform()[1]
             + y * ds.GetGeoTransform()[2],
             ds.GetGeoTransform()[3]
             + x * ds.GetGeoTransform()[4]
             + y * ds.GetGeoTransform()[5])
            for (x, y) in [(0, 0), (0, ds.RasterYSize), (ds.RasterXSize, 0),
                           (ds.RasterXSize, ds.RasterYSize)])

        # Read spatial reference as a osr.SpatialReference object or None
        # if there is no srs in metadata
        self.meta['srs'] = osr.SpatialReference(ds.GetProjection()) \
            if ds.GetProjection() \
            else None

        # Close file
        ds = None

    def block_windows(self, block_size=None):
        """Returns a list of block windows of the given size for the raster.

        It takes care of adjusting the size at right and bottom of the raster.

        :param block_size: wanted size for the blocks (defaults to the "natural"
                           block size of the raster
        :type block_size: tuple (xsize, ysize)
        :rtype: list of tuples in the form (i, j, xsize, ysize)
        """
        # Default size for blocks
        xsize, ysize = block_size \
            if block_size is not None \
            else self.meta['block_size']

        # Compute the list
        win_list = []
        for i in range(0, self.meta['height'], ysize):
            # Block height is ysize except at the bottom of the raster
            number_rows = ysize \
                if i + ysize < self.meta['height'] \
                else self.meta['height'] - i
            for j in range(0, self.meta['width'], xsize):
                # Block width is xsize except at the right of the raster
                number_cols = xsize \
                    if j + ysize < self.meta['width'] \
                    else self.meta['width'] - j
                win_list.append((j, i, number_cols, number_rows))
        return win_list

    def array(self, idx_band=None, block_win=None):
        """Returns the NumPy array corresponding to the raster.

        If the idx_band parameter is specified, then returns the NumPy array
        corresponding only to the band in the raster which has the given index.

        If the block_win parameter is specified, then returns the NumPy array
        corresponding to the block in the raster at the given window.

        :param idx_band: index of a band in the raster
        :type idx_band: int
        :param block_win: block window in the raster (x, y, hsize, vsize)
        :type block_win: 4-tuple of int
        :rtype: numpy.ndarray
        """
        # TODO: improve to support slice and range number of bands
        # Array size
        (hsize, vsize) = (block_win[2], block_win[3]) \
            if block_win is not None \
            else (self.meta['width'], self.meta['height'])
        depth = 1 if idx_band is not None else self.meta['count']

        # Initialize an empty array of correct size and type
        if depth != 1:
            array = np.empty((vsize, hsize, depth),
                             dtype=self.meta['dtype'].numpy_dtype)

        # Fill the array
        ds = gdal.Open(self.filename, gdal.GA_ReadOnly)
        for array_i in range(depth):
            ds_i = idx_band if idx_band is not None else array_i + 1
            if depth == 1 and block_win is not None:
                array = ds.GetRasterBand(ds_i).ReadAsArray(*block_win)
            elif depth == 1 and block_win is None:
                array = ds.GetRasterBand(ds_i).ReadAsArray()
            elif depth != 1 and block_win is not None:
                array[:, :, array_i] = ds.GetRasterBand(ds_i).ReadAsArray(
                    *block_win)
            else:
                array[:, :, array_i] = ds.GetRasterBand(ds_i).ReadAsArray()
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

    def set_nodata_value(self, value):
        """Writes the given nodata value into each raster's band.

        :param value: value to be used as nodata for the whole raster
        :type value: float
        """
        ds = gdal.Open(self.filename, gdal.GA_Update)
        for i in range(self.meta['count']):
            ds.GetRasterBand(i+1).SetNoDataValue(value)
        ds = None
        self.meta['nodata_value'] = value

    def remove_band(self, idx, out_filename):
        """Writes a new raster (in the specified output file) which is the same
        than the current raster, except that the band(s) at the given index has
        been remove.

        :param idx: list of index of the band(s) to remove (starts at 1)
        :type idx: list of int
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
                     if i + 1 not in idx]
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

    def rescale(self, nband, outmin, outmax, outype, out_filename):
        """Rescale a raster's band.

        :param nband: index value to the band to rescale
        :type nband: int
        :param outmin: minimum value to the rescaled band
        :type outmin: float
        :param outmax: maximum value to the rescaled band
        :type outmax: float
        :param outype: type of the output image (gdal type: gdal.GDT_UInt16,
                       gdal.GDT_UInt32, etc.)
        :type outype: float
        :param out_filename: path to the output file
        :type out_filename: str
        :returns: the ``Raster`` instance corresponding to the output file
        """
        ds = gdal.Open(self.filename, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(nband)
        band.ComputeStatistics(True)
        minval = band.GetMinimum()
        maxval = band.GetMaximum()
        nodata = band.GetNoDataValue()
        data = band.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
        indices = np.where(data != nodata)
        data[indices] = outmin + ((outmax - outmin)
                                  * ((data[indices] - minval)
                                     / (maxval - minval)))
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(out_filename, ds.RasterXSize, ds.RasterYSize,
                               self.meta['count'], outype)
        for i in range(self.meta['count']):
            if i + 1 != nband:
                dst_ds.GetRasterBand(i + 1).WriteArray(
                    ds.GetRasterBand(i + 1).ReadAsArray())
            else:
                dst_ds.GetRasterBand(i + 1).WriteArray(data)
            dst_ds.GetRasterBand(i + 1).ComputeStatistics(True)

        dst_ds.SetProjection(ds.GetProjection())
        dst_ds.SetGeoTransform(ds.GetGeoTransform())

        driver = None
        dst_ds = None
        ds = None

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

        out_raster = Raster(out_filename)
        if self.meta['datetime'] is not None:
            out_raster.set_datetime(self.meta['datetime'])

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

        out_raster = Raster(out_filename)
        if self.meta['datetime'] is not None:
            out_raster.set_datetime(self.meta['datetime'])

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

        out_raster = Raster(out_filename)
        if self.meta['datetime'] is not None:
            out_raster.set_datetime(self.meta['datetime'])

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

    def apply_mask(self, mask_raster, in_mask_value, out_filename,
                   out_mask_value = 65636):
        """Apply a mask to an image. It can be a multi-band image. It returns
        a raster object of the masked image.
        :param mask_raster: the raster object of the mask to apply
        :param in_mask_value: the value of the pixels "masked" in mask_raster
        :param out_filename: path of the output file
        :param out_mask_value: the value to set to the pixels masked in the
                            output file
        """

        #get the number of bands
        d = self.meta['count']

        #initialize variables
        list_raster = []
        list_file  = []
        rasters = [self.filename, mask_raster.filename]
        BandMath = otb.Registry.CreateApplication("BandMath")

        #for each band, apply the mask and create a file
        for i in range(d):
            #conditional expression to set a new value to the input band
            exp = "(im2b1 == {}) ? {} : im1b{}".format(in_mask_value,
                                                        out_mask_value,i+1)

            out = os.path.join(gettempdir(),'mono_mask_{}.tif'.format(i))
            BandMath.SetParameterStringList("il", rasters)
            BandMath.SetParameterString("out", out)
            BandMath.SetParameterString("exp", exp)
            BandMath.ExecuteAndWriteOutput()
            #store the temp files path and corresponding raster objects
            list_raster.append(Raster(out))
            list_file.append(out)

        #Concatenate each mono-band file in one file
        concatenate_images(list_raster, out_filename)

        #Delete the temp files
        for fi in list_file:
            os.remove(fi)
        out_raster = Raster(out_filename)
        out_raster.set_nodata_value(out_mask_value)

        return out_raster

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

            print "vectorisation step has been realized succesfully"

    def get_stat(self,
                 orig_raster,
                 out_filename,
                 stats=["mean", "std", "min", "max", "per"],
                 percentile=[20, 40, 50, 60, 80],
                 ext="Gtiff"):
        """Calcul statistics of the labels from a label image and raster. The
        statistics calculated by default are : mean, standard deviation, min,
        max and the 20, 40, 50, 60, 80th percentiles. The output is an image
        at the given format that contains n_band * n_stat_features bands. This
        method uses the GDAL et NUMPY library.

        :param orig_raster: The raster object on which the statistics are
                            calculated
        :param out_filename: Path of the output image.
        :param stats: List of the statistics to be calculated. By default, all
                      the features are calculated,i.e. mean, std, min, max and
                      per.
        :param percentile: List of the percentile to be calculated. By
                           default, the percentiles are 20, 40, 50, 60, 80.
        :param ext: Format in wich the output image is written. Any formats
                    supported by GDAL
        """

        # load thes images
        data = gdal.Open(orig_raster.filename)
        #TODO : remove the following lines if necessary
        #mask = gdal.Open(mask_raster.filename)
        #M = mask.GetRasterBand(1).ReadAsArray()

        # Get some parameters
        nx = data.RasterXSize
        ny = data.RasterYSize
        d = data.RasterCount
        GeoTransform = data.GetGeoTransform()
        Projection = data.GetProjection()

        # load the label file and sort his values
        label = gdal.Open(self.filename)
        L = label.GetRasterBand(1).ReadAsArray()
        L_sorted = np.unique(L)

        # calcul the number of stats to be calculated per type (percentile or
        # others)
        if "per" in stats:
            len_per = len(percentile)  # number of percentiles
            len_var = len(stats) - 1   # number of other stats
        else:
            len_per = 0
            len_var = len(stats)
        nb_var = len_per + len_var
        d_obj = d * nb_var  # number of band in the object

        # creation of a function dictionnary
        fn = {}
        fn["mean"] = np.mean
        fn["std"] = np.std
        fn["min"] = np.min
        fn["max"] = np.max
        fn["per"] = np.percentile

        # Initialization of the array that will received each band iteratively
        im = np.empty((ny, nx))

        # Create the object file
        driver = gdal.GetDriverByName(ext)
        output = driver.Create(out_filename, nx, ny, d_obj, gdal.GDT_Float64)
        output.SetGeoTransform(GeoTransform)
        output.SetProjection(Projection)

        #TODO : créer une liste unique de type de stat
        #Compute the object image
        for j in range(d): #for each band
            im = data.GetRasterBand(j+1).ReadAsArray()#load the band in a array
            for k in range(nb_var):#for each stat
                obj = np.empty((ny,nx))
                if k < len_var: #if this is not a percentile
                    name = stats[k]
                    arg = [""]
                else:
                    name = "per"
                    arg = ["", percentile[k - len_var]]
                for i in L_sorted:  # for each label
                    t = np.where((L == i))
                    arg[0] = im[t[0], t[1]]
                    obj[t[0], t[1]] = fn[name](*arg)
                # Write the new band
                temp = output.GetRasterBand(j * nb_var + k + 1)
                temp.WriteArray(obj[:, :])
                temp.FlushCache()

        # Close the files
        label = None
        data = None
        output = None
