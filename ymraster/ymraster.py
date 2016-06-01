# -*- coding: utf-8 -*-

"""The `ymraster` module encloses the main methods and functions to work with
raster images.
"""
import os 
os.environ["ITK_AUTOLOAD_PATH"] = '/usr/lib/otb/applications'
os.environ["PYTHONPATH"] = '/usr/lib/otb/python'
import sys
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
    from osgeo import osr, gdal, ogr
    gdal.UseExceptions()
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install GDAL.")

try:
    import numpy as np
    import numpy.ma as ma
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install NumPy.")
        
try:
    from rasterstats import zonal_stats
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install rasterstats.")   

from raster_dtype import RasterDataType
from driver_ext import DriverExt
import array_stat

from fix_proj_decorator import fix_missing_proj

from collections import Sized
from datetime import datetime
from time import mktime
import os, csv
import shutil
from tempfile import gettempdir


try:
    from netCDF4 import Dataset, stringtoarr
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install netCDF4 library.")   

from xml.dom import minidom


def _dt2float(dt):
    """Returns a float corresponding to the given datetime object.

    :param dt: datetime to convert into a float
    :type dt: datetime.datetime
    :rtype: float
    """
    return mktime(dt.timetuple())
    
def manage_clouds(block, date):
    """Extract a table of pixel like the number of pixel without cloud.
    
    :param block: a table of pixel value
    :type block: tuple of number (x, y, xsize, ysize)
    """
    
    block_cloud = np.greater(block, 0) # if block_stack != 0 then True else False
    account_cloud = np.sum(block_cloud[0], axis=1) # Account to block_stack if != 0 then append +1 per rows
#    print account_cloud
    
    # Remove cloud date
    clear_clouddate = np.choose(block_cloud[0], (' ', date[0]))
#    print clear_clouddate

    return account_cloud, clear_clouddate


#def CsvStock(In, full_data):
#    
#    # Ecriture dans un CSV
#    Outfile = In.split('.')[0] + '.csv'
#    if os.path.exists(Outfile):
#        os.remove(Outfile)
#    Out = open(Outfile,  "wb")
#    Outcsv = csv.writer(Out, delimiter=';')
#    
#    for i in range(len(full_data)):
##        for j in range(len(Stats[i])):
#    Outcsv.writerow(full_data[i])
#    
#    # Referme le csv de sortie
#    Out.close()
#    
#    return Outfile


def write_file(out_filename, array=None, overwrite=False,
               xoffset=0, yoffset=0, band_idx=1, **kw):
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
    :param array: the NumPy array to save. None means that an empty file will be
                  created or, if file exists, that no data will be written
                  (except metadata if specified)
    :type array: np.ndarray
    :param overwrite: if True, overwrite file if exists. False by default.
    :type overwrite: bool
    :param xoffset: horizontal offset. First index from which to write the array
                    in the output file if the array is smaller (default: 0)
    :type xoffset: float
    :param yoffset: Vertical offset. First index from which to write the array
                    in the output file if the array is smaller (default: 0)
    :type yoffset: float
    :param band_idx: depth offset. First band from which to write the array
                    in the output file if the array is smaller (default: 1)
    :type band_idx: int
    :param width: horizontal size of the image to be created
    :type width: int
    :param height: vertical size of the image to be created
    :type width: int
    :param depth: number of bands of the image to be created
    :type depth: int
    :param dtype: data type to use for the output file. None means that the file
                  already exists
    :type dtype: RasterDataType
    :param date_time: date/time to write in the output file metadata
    :type date_time: datetime.datetime
    :param nodata_value: value to set as NODATA in the output file metadata
    :type nodata_value: dtype
    :param srs: projection to write in the output file metadata
    :type srs: osr.SpatialReference
    :param transform: geo-transformation to use for the output file
    :type transform: 6-tuple of floats
    """
    # Size & data type of output image
    xsize, ysize = (kw['width'], kw['height']) \
        if kw.get('width') and kw.get('height') \
        else (array.shape[1], array.shape[0])
    try:
        number_bands = kw['count'] \
            if kw.get('count') \
            else array.shape[2]
    except IndexError:
        number_bands = 1
    dtype = kw['dtype'] \
        if kw.get('dtype') \
        else RasterDataType(numpy_dtype=array.dtype.type)

    # Create an empty raster file if it does not exists or if overwrite is True
    try:
        assert not overwrite
        out_ds = gdal.Open(out_filename, gdal.GA_Update)
    except (AssertionError, RuntimeError):
        
        _, ext = os.path.splitext(out_filename)
        driver = DriverExt(extension=ext).gdal_driver
        out_ds = driver.Create(out_filename,
                               xsize,
                               ysize,
                               number_bands,
                               dtype.gdal_dtype)
                               
    # Set metadata
    if kw.get('date_time'):
        out_ds.SetMetadata(
            {'TIFFTAG_DATETIME': kw['date_time'].strftime('%Y:%m:%d %H:%M:%S')})
    if kw.get('srs'):
        out_ds.SetProjection(kw['srs'].ExportToWkt())
    if kw.get('transform'):
        out_ds.SetGeoTransform(kw['transform'])

    # If no array, nothing else to do
    if array is None:
        return

    # Save array at specified band and offset
    for i in range(number_bands):
        band = out_ds.GetRasterBand(i+band_idx)
        if number_bands == 1:
            band.WriteArray(array, xoff=xoffset, yoff=yoffset)
        else:
            band.WriteArray(array[:, :, i], xoff=xoffset, yoff=yoffset)
        band.FlushCache()
    band = None
    out_ds = None


def concatenate_rasters(*rasters, **kw):
    """Write a raster which is the concatenation of the given rasters, in order.

    All bands in all input rasters must have same size.

    Moreover, this function is quite conservative about projections: all bands
    should have the same projection and same extent

    Finally, if data types are different, then everything will be converted to
    the default data type in OTB (_float_ currently).

    :param rasters: the rasters to concatenate
    :type rasters: list of `Raster` instances
    :param out_filename: path to the output file. If omitted, the append all
                         rasters into the first one given
    :type out_filename: str to the output file
    """
    # Check for proj, extent & type (and that list not empty)
    rasters = list(rasters)
    raster0 = rasters[0]
    srs, extent, otb_dtype = (raster0.srs,
                              raster0.gdal_extent,
                              raster0.dtype.otb_dtype)
    assert srs, \
        "Image has no Coordinate Reference System: '{:f}'".format(raster0)
    same_type = True
    for raster in rasters:
        assert raster.srs \
            and raster.srs.IsSame(srs), \
            "Images have not the same Coordinate Reference System: "
        "'{:f}' and '{:f}'".format(raster0, raster)
        assert raster.gdal_extent == extent, \
            "Images have not the same extent: "
        "'{:f}' and '{:f}'".format(raster0, raster)
        if raster.dtype.otb_dtype != otb_dtype:
            same_type = False

    # Out file
    out_filename = kw['out_filename'] \
        if kw.get('out_filename') \
        else os.path.join(gettempdir(), 'concat.tif')

    # Perform the concatenation
    filenames = [raster.filename for raster in rasters]
    ConcatenateImages = otb.Registry.CreateApplication("ConcatenateImages")
    ConcatenateImages.SetParameterStringList("il", filenames)
    ConcatenateImages.SetParameterString("out", out_filename)
    if same_type:
        ConcatenateImages.SetParameterOutputImagePixelType("out", otb_dtype)
    ConcatenateImages.ExecuteAndWriteOutput()

    # Overwrite if needed
    if not kw.get('out_filename'):
        shutil.copy(out_filename, raster0.filename)
        os.remove(out_filename)

def nbpxl_clear(*rasters, **kw):
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
    :type rasters: list of `Raster` instances
    :param band_idx: index of the band to compute statistics on (default: 1)
    :type band_idx: int
    :param stats: list of stats to compute
    :type stats: list of str
    :param time_raster: T0 raster (e.g. Snow brake date)
    :type time_raster: raster
    :param date2float: function which returns a float from a datetime object.
                       By default, it is the time.mktime() function
    :type date2float: function
    :param out_filename: path to the output file. If omitted, the filename is
                         based on the stats to compute.
    :type out_filename: str
    
    : param in_cloud: if the cloud's layers exist, it's necessary to give the name's 
                         cloud images (By default: no exist).
    : type in_cloud: list of `Raster`
    """
    
    # Presence of the cloud's image
    inclouds = kw['in_cloud'] \
        if kw.get('in_cloud') \
        else None
    
    # Band to compute statistics on
    band_idx = kw['band_idx'] \
        if kw.get('band_idx') \
        else 1

    # Out filename
    out_filename = kw['out_filename'] \
        if kw.get('out_filename') \
        else '{}.tif'.format('_'.join('nbclear_pxl'))
    
    # Number of bands in output file (1 for each stat +1 for each summary stat + n for number of days between T0 (snow break date) and max) + cloud's layer (=1)
#    depth = len(stats) + len([statname for statname in stats
#                              if array_stat.ArrayStat(statname).is_summary]) * 2 + 1
    depth=1 # Number of bands in output file to cloud's layer

    # Pré-ouverture pour écrire dans un CSV
    Outfile = out_filename.split('.')[0] + '.csv'
    if os.path.exists(Outfile):
        os.remove(Outfile)
    Out = open(Outfile,  "wb")
    Outcsv = csv.writer(Out, delimiter=';')    

    # Create an empty file of correct size and type
    rasters = list(rasters)
    raster0 = rasters[0]
    meta = raster0.meta
#    print meta
    meta['count'] = depth
#    meta['dtype'] = RasterDataType(lstr_dtype='float64') # To multi-spectral raster
    meta['dtype'] = RasterDataType(lstr_dtype='int16') 
    write_file(out_filename, overwrite=True, **meta)
    
    # TODO: improve to find better "natural" blocks than using the "natural"
    # segmentation of simply the first image
    for block_win in raster0.block_windows():
        # Turn each block into an array and concatenate them into a stack
        if inclouds is not None :# if the cloud's raster define
            block_arrays = [raster.array_from_bands(band_idx, block_win=block_win, \
#                            cloud_image=Raster(raster.filename[:-4] + '_NUA' + raster.filename[-4:]))
                            cloud_image = Raster([incloud.filename for incloud in inclouds if incloud.date_time == raster.date_time][0]))
                            for raster in rasters]
#            date_block = [raster.date_time for raster in rasters]
        else : # Else launch this command without the cloud's raster
            block_arrays = [raster.array_from_bands(band_idx, block_win=block_win)
                            for raster in rasters]
        
        # Matrix of raster coordinates pixel
        array_x, array_y =raster0.coord_pxl(block_win=block_win)
        
        block_stack = np.dstack(block_arrays) \
            if len(block_arrays) > 1 \
            else block_arrays[0]
        
        # date_arrays, matrix with raster's date
        date_arrays = [np.chararray((len(block_arrays[0]), len(block_arrays[0][0])), itemsize=19) for raster in rasters]        
        for r in range(len(rasters)):
            date_arrays[r][:] = str(rasters[r].date_time) + ' '

        date_stack = np.dstack(date_arrays) \
            if len(date_arrays) > 1 \
            else date_arrays[0]            
        
        # Compute each stat for the block and append the result to a list
        stat_array_list = []  
        
        # Add cloud's raster in final layer stack
        if incloud is not None:
            account_cloud, clear_clouddate = manage_clouds(block_stack, date_stack)
            stat_array_list.append(account_cloud)
            
        # Insert coordinates x and y before date data without cloud
        # Add date data in a database 
        bd_cloud = []  
        for cl in range(len(clear_clouddate)):
            bd_cloud.append(np.insert(np.insert(np.insert(clear_clouddate[cl], 0, account_cloud[cl]), 0, array_y[0][cl][0]), 0, array_x[0][cl][0]))
            # Ecriture dans le csv
            Outcsv.writerow(bd_cloud[cl])       
        
        # Concatenate results into a stack and save the block to the output file
        stat_stack = np.dstack(stat_array_list) \
            if len(stat_array_list) > 1 \
            else np.array([stat_array_list[0]], dtype=np.dtype(np.int16))

        xoffset, yoffset = block_win[0], block_win[1]
        write_file(out_filename, array=stat_stack,
                   xoffset=xoffset, yoffset=yoffset)
  
    # Referme le csv de sortie
    Out.close()


def temporal_stats(*rasters, **kw):
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
    :type rasters: list of `Raster` instances
    :param band_idx: index of the band to compute statistics on (default: 1)
    :type band_idx: int
    :param stats: list of stats to compute
    :type stats: list of str
    :param time_raster: T0 raster (e.g. Snow brake date)
    :type time_raster: raster
    :param date2float: function which returns a float from a datetime object.
                       By default, it is the time.mktime() function
    :type date2float: function
    :param out_filename: path to the output file. If omitted, the filename is
                         based on the stats to compute.
    :type out_filename: str
    
    : param in_cloud: if the cloud's layers exist, it's necessary to give the name's 
                         cloud images (By default: no exist).
    : type in_cloud: list of `Raster`
    """
    
    # Stats to compute
    stats = kw['stats'] \
        if kw.get('stats') \
        else ['min', 'max', 'mean']

    # Date function
    date2float = kw['date2float'] \
        if kw.get('date2float') \
        else _dt2float
        
    # Raster T0
    t0 = kw['time_raster'] \
        if kw.get('time_raster') \
        else 1
    
    # Presence of the cloud's image
    inclouds = kw['in_cloud'] \
        if kw.get('in_cloud') \
        else None
    
    # Band to compute statistics on
    band_idx = kw['band_idx'] \
        if kw.get('band_idx') \
        else 1

    # Out filename
    out_filename = kw['out_filename'] \
        if kw.get('out_filename') \
        else '{}.tif'.format('_'.join(stats))

    # Number of bands in output file (1 for each stat +1 for each summary stat + n for number of days between T0 (snow break date) and max) + cloud's layer (=1)
    depth = len(stats) + len([statname for statname in stats
                              if array_stat.ArrayStat(statname).is_summary]) * 2 + 1

    # Create an empty file of correct size and type
    rasters = list(rasters)
    raster0 = rasters[0]
    meta = raster0.meta
    meta['count'] = depth
    meta['dtype'] = RasterDataType(lstr_dtype='float64')
    write_file(out_filename, overwrite=True, **meta)

    # TODO : Manage Clouds
    # Recuperer le fichier de masquage des nuages avec le nom de fichiers
    # Attribuer une valeur "nodata" pour qu'elles ne soient pas prise en compte dans l'étude temporelle
    # Valeur "0" du masque nuage correspond à la donnée utilisable
    a = 0
    
    # TODO: improve to find better "natural" blocks than using the "natural"
    # segmentation of simply the first image
    for block_win in raster0.block_windows():
        # Turn each block into an array and concatenate them into a stack
        if inclouds is not None :# if the cloud's raster define
            block_arrays = [raster.array_from_bands(band_idx, block_win=block_win, \
#                            cloud_image=Raster(raster.filename[:-4] + '_NUA' + raster.filename[-4:]))
                            cloud_image = Raster([incloud.filename for incloud in inclouds if incloud.date_time == raster.date_time][0]))
                            for raster in rasters]
        else : # Else launch this command without the cloud's raster
            block_arrays = [raster.array_from_bands(band_idx, block_win=block_win)
                            for raster in rasters]

        # block t0 raster management
        block_t0 = t0.array_from_bands(1,block_win=block_win) \
            if t0 != 1 \
            else 1
        
        block_stack = np.dstack(block_arrays) \
            if len(block_arrays) > 1 \
            else block_arrays[0]
            
        # TODO : Module de lissage de Maylis
        a = a+1
        print a
        # Compute each stat for the block and append the result to a list
        stat_array_list = []
        for statname in stats:
            astat = array_stat.ArrayStat(statname, axis=2)
            stat_array_list.append(astat.compute(block_stack))
            if astat.is_summary:
                # If summary stat, compute date of occurence
                # indice de la date dans la série temporelle en fonction de la date de l'image par ex : 0 = 25/04/2013                
                date_array = astat.indices(block_stack)
                for x in np.nditer(date_array, op_flags=['readwrite']):
                    try:
                        x[...] = date2float(rasters[x].date_time)
                    except TypeError:
                        raise ValueError(
                            'Image has no date/time metadata: {:f}'.format(
                                rasters[x]))
                stat_array_list.append(date_array)
                
                # T0 management    
                if isinstance(block_t0, type(block_arrays)):
                    # Calcul du nombre de jours entre le max et le t0                              
                    # Mask value equal to 0 for t0 raster (no data)                    
                    mask_no_snow = block_t0 == 0
                    block_t0_masked = np.ma.array(block_t0, mask=mask_no_snow)
                    # Nb days between date of max/min and T0
                    nb_days = np.abs(date_array - block_t0_masked)/ (3600 * 24)
                    # Add raster of nb days in stats raster
                    stat_array_list.append(nb_days.astype(int))        

        # Add cloud's raster in final layer stack
        if incloud is not None:
            stat_array_list.append(manage_clouds(block_stack))
        
        # Concatenate results into a stack and save the block to the output file
        stat_stack = np.dstack(stat_array_list) \
            if len(stat_array_list) > 1 \
            else stat_array_list[0]

        xoffset, yoffset = block_win[0], block_win[1]
        write_file(out_filename, array=stat_stack,
                   xoffset=xoffset, yoffset=yoffset)

def mosaic(*rasters, **kw):
    
    # prefixe
    prefixe = kw['prefixe'] \
        if kw.get('prefixe') \
        else ''
    
    # List of rasters 
    rasters = list(rasters)
    
    
    # List of raster's date
    dates_arrays = [raster._date_time.date() for raster in rasters]
    
    # Remove double date
    date_arrays = np.unique(dates_arrays)

    # Select raster to create a mosaic
    for raster_date in date_arrays:
        # Keep raster to mosaic in a table
        raster_mosaic = [raster for raster in rasters if raster._date_time.date() == raster_date]
        print 'Il y a ' + str(len(raster_mosaic)) + ' image(s)'    

        # Create a new extend to the mosaic layer
        X_moins = 0
        X_plus = 0
        Y_moins = 0
        Y_plus = 0

        
        for raster in raster_mosaic :
            for i in range(len(raster.gdal_extent)):
                if raster.gdal_extent[i][0] < X_moins or X_moins == 0:
                    X_moins = raster.gdal_extent[i][0]
                if raster.gdal_extent[i][0] > X_plus :
                    X_plus = raster.gdal_extent[i][0]
                
                if raster.gdal_extent[i][1] < Y_moins  or Y_moins == 0:
                    Y_moins = raster.gdal_extent[i][1]
                if raster.gdal_extent[i][1] > Y_plus :
                    Y_plus = raster.gdal_extent[i][1]
        
        new_extent = tuple()
        new_extent = tuple([(X_moins, Y_plus), (X_moins, Y_moins), \
                                        (X_plus, Y_plus), (X_plus, Y_moins)])# (UL, DL, UR, DR)
        # U : Upper ... D : Down ... L : Left ... R : Right
        
#        new_extent = tuple([raster_mosaic[len(raster_mosaic)-1].gdal_extent[0], \
#                                        raster_mosaic[0].gdal_extent[1], \
#                                        raster_mosaic[len(raster_mosaic)-1].gdal_extent[2], \
#                                        raster_mosaic[0].gdal_extent[3]]) # (UL, DL, UR, DR)
        # U : Upper ... D : Down ... L : Left ... R : Right
        
        # Create a variable empty with the dimensions of new extend mosaic layer
        cols = (new_extent[0][1] - new_extent[1][1]) / raster_mosaic[0].transform[1]
        rows = (new_extent[2][0] - new_extent[0][0]) / raster_mosaic[0].transform[1]
        depth = raster_mosaic[0].count
        
        array = np.empty((cols, rows, depth), dtype=raster_mosaic[0].dtype.numpy_dtype)
        
        # Create a full table of the layer out
        raster_mosaic.reverse()
        for raster in  raster_mosaic :
            print raster.filename
            x_pxl = abs(new_extent[0][0] - raster.transform[0]) / raster.transform[1]
            y_pxl = abs(new_extent[0][1] - raster.transform[3]) / raster.transform[1]

            # Return no data value
            raster.nodata_value = -10000
            
            try:
                array[y_pxl:y_pxl+raster.height, \
                        x_pxl:x_pxl+raster.width, \
                        0:raster.count] = raster.array_from_bands()
            except:
                # For images with only band. So we need to reshape the table like this.
                # Because for python (50,50,1) != (50,50)
                array[y_pxl:y_pxl+raster.height, \
                        x_pxl:x_pxl+raster.width, \
                        0:raster.count] = np.reshape(raster.array_from_bands(), (raster.array_from_bands().shape[0], \
                                                                                                                        raster.array_from_bands().shape[1], depth))
            
        # Name of mosaic image
        in_name = raster_mosaic[0].filename
        outfile = in_name.split('-')[0] + '_' + prefixe + '.tif'
        print 'Ecriture de la mosaïque ' + outfile
        # Image's metadata
        meta = raster.meta
        meta['gdal_extent'] = new_extent
        meta['transform'] = tuple([new_extent[0][0], meta['transform'][1], meta['transform'][2], \
                                                        new_extent[0][1], meta['transform'][4], meta['transform'][5]])
        meta['height'] = int(cols)
        meta['width'] = int(rows)
        meta['block_size'] = tuple([int(cols), 1])

        write_file(outfile, overwrite=True, **meta)
        # Writing in the outfile
        write_file(outfile, array=array, overwrite=True, **meta)

#    return
def clip(*rasters, **kw):
    
    # prefixe
    prefixe = kw['prefixe'] \
        if kw.get('prefixe') \
        else ''
    # vector
    vector = kw['vector']
        
    # List of rasters 
    rasters = list(rasters)    
    for raster in rasters:
        outfile = raster.filename.split('.')[0] + '_' + prefixe + '.tif'
        print outfile
        os.system('gdalwarp -dstnodata -10000 -q -cutline ' + vector + ' -crop_to_cutline -of GTiff ' + raster.filename + ' ' + outfile)
        
        
class Raster(Sized):
    """Represents a raster image that was read from a file.

    The whole raster *is not* loaded into memory. Instead this class records
    useful information about the raster (number and size of bands, projection,
    etc.) and provide useful methods for comparing rasters, computing some
    indices, etc.
    """

    def __init__(self, filename):
        """Create a new `YmRaster` instance from an image file.

        Parameters
        ----------
        filename : str
            path to the image file to read
        """
        self._filename = filename
        self.refresh()
        
        """ Poucentage of cloud in the layer and by block"""
        self._block_cloud = []
        self._cloud = 0

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__,
                                 os.path.abspath(self._filename))

    def __str__(self):
        return self.__format__()

    def __format__(self, format_spec=''):
        if format_spec and format_spec.startswith('f'):
            s = os.path.basename(self._filename)
            return s.__format__(format_spec[1:])
        elif format_spec and format_spec.startswith('b'):
            s = os.path.basename(os.path.splitext(self._filename)[0])
            return s.__format__(format_spec[1:])
        else:
            s = os.path.abspath(self._filename)
            return s.__format__(format_spec)

    def __len__(self):
        return self._count
    
    @property
    def filename(self):
        """The raster's filename (str)"""
        return self._filename

    @property
    def driver(self):
        """The raster's GDAL driver (DriverExt object)"""
        return self._driver

    @property
    def width(self):
        """The raster's width (int)"""
        return self._width

    @property
    def height(self):
        """The raster's height (int)"""
        return self._height

    @property
    def count(self):
        """The raster's count or number of bands (int)"""
        return self._count

    @property
    def dtype(self):
        """The raster's data type (RasterDataType object)"""
        return self._dtype

    @property
    def block_size(self):
        """The raster's natural block size (tuple of int)"""
        return self._block_size

    @property
    def date_time(self):
        """The raster's date/time (datetime.datetime object)"""
        return self._date_time

    @property
    def nodata_value(self):
        """The raster's NODATA value (self.dtype value)"""
        return self._nodata_value

    @property
    def transform(self):
        """The raster's geo-transformation (tuple of floats)"""
        return self._transform

    @property
    def gdal_extent(self):
        """The raster's extent, as given by GDAL (tuple of floats)"""
        return self._gdal_extent

    @property
    def srs(self):
        """The raster's projection (osr.SpatialReference object)"""
        return self._srs
        
    @property
    def meta(self):
        """Returns a dictionary containing the raster's metadata

        :rtype: dict
        """
        return {k: getattr(self, k)
                for k, v in self.__class__.__dict__.iteritems()
                if k != 'meta' and isinstance(v, property)}

    @date_time.setter
    def date_time(self, dt):
        ds = gdal.Open(self._filename, gdal.GA_Update)
        ds.SetMetadata({'TIFFTAG_DATETIME': dt.strftime('%Y:%m:%d %H:%M:%S')})
        ds = None
        self.refresh()

    @nodata_value.setter
    def nodata_value(self, value):
        ds = gdal.Open(self._filename, gdal.GA_Update)
        for i in range(self._count):
            ds.GetRasterBand(i+1).SetNoDataValue(value)
        ds = None
        self.refresh()

    @srs.setter
    def srs(self, sr):
        ds = gdal.Open(self._filename, gdal.GA_Update)
        ds.SetProjection(sr.ExportToWkt())
        ds = None
        self.refresh()

    def refresh(self):
        """Reread the raster's properties from file."""
        ds = gdal.Open(self._filename, gdal.GA_ReadOnly)
        self._driver = DriverExt(gdal_driver=ds.GetDriver())
        self._width = ds.RasterXSize
        self._height = ds.RasterYSize
        self._count = ds.RasterCount
        self._dtype = RasterDataType(
            gdal_dtype=ds.GetRasterBand(1).DataType)
        self._block_size = tuple(
            ds.GetRasterBand(1).GetBlockSize())
        try:
            self._date_time = datetime.strptime(
                ds.GetMetadataItem('TIFFTAG_DATETIME'), '%Y:%m:%d %H:%M:%S')
        except ValueError:  # string has wrong datetime format
            self._date_time = None
        except TypeError:   # there is no DATETIME tag
            self._date_time = None
        self._nodata_value = ds.GetRasterBand(1).GetNoDataValue()
        self._transform = ds.GetGeoTransform(can_return_null=True)
        self._gdal_extent = tuple(
            (self._transform[0]
             + x * self._transform[1]
             + y * self._transform[2],
             self._transform[3]
             + x * self._transform[4]
             + y * self._transform[5])
            for (x, y) in ((0, 0), (0, ds.RasterYSize), (ds.RasterXSize, 0),
                           (ds.RasterXSize, ds.RasterYSize)))
        self._srs = osr.SpatialReference(ds.GetProjection()) \
            if ds.GetProjection() \
            else None

        # Close file
        ds = None

    def has_same_extent(self, raster, prec=0.01):
        """Returns True if the raster and the given one has same extent.

        Parameters
        ----------
        raster : `Raster`
            raster to compare extent with.
        prec : float, optional
            difference threshold under which coordinates are said to be equal
            (default: 1).

        Returns
        -------
        bool
            boolean saying if both rasters have same extent.

        Examples
        --------
        >>> filename = os.path.join('data', 'l8_20130425.tif')
        >>> other_filename = os.path.join('data', 'l8_20130714.tif')
        >>> raster = Raster(filename)
        >>> other_raster = Raster(other_filename)
        >>> raster.has_same_extent(other_raster)
        True
        """
        extents_almost_equals = tuple(
            map(lambda t1, t2: (abs(t1[0] - t2[0]) <= prec,
                                abs(t1[1] - t2[1]) <= prec),
                self._gdal_extent, raster.gdal_extent))
        return extents_almost_equals == ((True, True), (True, True),
                                         (True, True), (True, True))

    def block_windows(self, block_size=None):
        """Yield coordinates of each block in the raster, in order.

        It takes care of adjusting the block size at right and bottom edges.

        Parameters
        ----------
        block_size : tuple of int (xsize, ysize), optional
            wanted size for each block. By default, the "natural" block size of
            the raster is used.

        Yields
        ------
        tuple of int (i, j, xsize, ysize)
            coordinates of each block
        """
        # Default size for blocks
        xsize, ysize = block_size if block_size else self.block_size

        # Compute the next block window
        for i in range(0, self._height, ysize):
            # Block height is ysize except at the bottom of the raster
            number_rows = ysize \
                if i + ysize < self._height \
                else self._height - i
            for j in range(0, self._width, xsize):
                # Block width is xsize except at the right of the raster
                number_cols = xsize \
                    if j + ysize < self._width \
                    else self._width - j
                yield (j, i, number_cols, number_rows)


    def cloud(self, area_cloud):
        """Returns a pourcentage of cloud on whole of the layer
            by computing mean of cloud's pourcentage on every block
            
            Param: area_cloud (required) -> cloud's pourcentage on a block
            Type of area_cloud -> float
        """
        # Table of cloud's pourcentage on every block
        self._block_cloud.append(area_cloud)
        
        # Pourcentage of cloud on whole of the layer
        self._cloud = np.mean(self._block_cloud)
        
        return self._cloud
        
    def coord_pxl(self, **kw):
        """Returns a NumPy array from the raster of pixels coordinates.

        If the `block_win` parameter is given, then only values inside the
        specified coordinates (window) are returned.

        Parameters
        ----------
        block_win : tuple of int (x, y, xsize, ysize), optional
            block window to get array from. By default, all pixel values are
            returned.

        Returns
        -------
        numpy.ndarray
            array extracted from the raster.
        """
        # Get size of output array and initialize an empty array (if multi-band)
        (hsize, vsize) = (kw['block_win'][2], kw['block_win'][3]) \
            if kw.get('block_win') \
            else (self._width, self._height)
        array_x = np.zeros((vsize, hsize, 1), dtype=np.float64)
        array_y = np.zeros((vsize, hsize, 1), dtype=np.float64)
                
        for i in range(array_x.shape[0]):
            for j in range(array_x.shape[1]):
                # Move the window on i and block_win x (self._transform[1]*i)+(kw['block_win'][1]*self._transform[1])
                array_x[i][j] = self._transform[0]+self._transform[1]*(i+kw['block_win'][1])
                # Move the window on j and block_win y (self._transform[5]*j)+(kw['block_win'][0]*self._transform[5])
                array_y[i][j] = self._transform[3]+self._transform[5]*(j+kw['block_win'][0])
                
        return array_x, array_y
        
    
    def array_from_bands(self, *idxs, **kw):
        """Returns a NumPy array from the raster according to the given
        parameters.

        If some `idxs` are given, then only values from corresponding bands are
        returned.

        If the `block_win` parameter is given, then only values inside the
        specified coordinates (window) are returned.

        The two preceding parameters can be combined to get, for example, a
        block only from one band.

        If the `mask_nodata` parameter is given and `True`, then NODATA values
        are masked in the array and a `MaskedArray` is returned.

        Parameters
        ----------
        idxs : int, optional
            indices of bands to get array from. By default, all bands are
            returned.
        block_win : tuple of int (x, y, xsize, ysize), optional
            block window to get array from. By default, all pixel values are
            returned.
        mask_nodata : bool, optional
            if `True` NODATA values are masked in a returned `MaskedArray`.
            Else a simple `ndarray` is returned whith all values. True by
            default.
            
        cloud_image : Raster, optional
            if exist, the cloud on the raster will be remplaced by 0.

        Returns
        -------
        numpy.ndarray or numpy.ma.MaskedArray
            array extracted from the raster.
        """
        # Get size of output array and initialize an empty array (if multi-band)
        (hsize, vsize) = (kw['block_win'][2], kw['block_win'][3]) \
            if kw.get('block_win') \
            else (self._width, self._height)
        depth = len(idxs) if idxs else self._count
        if depth > 1:
            array = np.empty((vsize, hsize, depth),
                             dtype=self.dtype.numpy_dtype)

        # Fill the array
        ds = gdal.Open(self._filename, gdal.GA_ReadOnly)
        for array_i in range(depth):
            ds_i = idxs[array_i] if idxs else array_i + 1
            if depth == 1 and kw.get('block_win'):
                array = ds.GetRasterBand(ds_i).ReadAsArray(*kw['block_win'])
            elif depth == 1 and not kw.get('block_win'):
                array = ds.GetRasterBand(ds_i).ReadAsArray()
            elif depth > 1 and kw.get('block_win'):
                array[:, :, array_i] = ds.GetRasterBand(ds_i).ReadAsArray(
                    *kw['block_win'])
            else:
                array[:, :, array_i] = ds.GetRasterBand(ds_i).ReadAsArray()
        ds = None
        
        # Fill the cloud array if exist
        if kw.get('cloud_image'):
            ds = gdal.Open(kw.get('cloud_image')._filename, gdal.GA_ReadOnly)
            # There is a cloud's band
            if kw.get('block_win'):
                cloud_array = ds.GetRasterBand(1).ReadAsArray(*kw['block_win'])
            elif not kw.get('block_win'):
                cloud_array = ds.GetRasterBand(1).ReadAsArray()
            ds=None
            
            # Create a boolean's table with true if pixel ~ CN and false if pixel = -10000 or 0
            bool_area = np.in1d(array, [-10000, 0], invert=True)
            # Create a boolean's table with true if pixel is clear ie without cloud and with a spectral data
            bool_clear = np.choose(np.in1d(cloud_array, 0), (False, bool_area))
            # Pourcentage of cloud in the block
            try:
                pourc_cloud = 100 - ((float(np.sum(bool_clear)) / np.sum(bool_area)) * 100)
            except ZeroDivisionError:
                pourc_cloud = 100
            
            # Compute cloud's area in pourcentage
            self.cloud(area_cloud=pourc_cloud)
            
            # Cloud mask boolean
            cloud_mask = np.greater(cloud_array, 0) # if cloud_array != 0 then True else False
            # Raster without cloud
            array = np.choose(cloud_mask, (array, 0)) # if cloud_mask = True then 0 else raster
            
            
        # Returned a masked array if wanted or if no indication
        
        if kw.get('mask_nodata') or 'mask_nodata' not in kw:
            return ma.masked_where(array == self._nodata_value, array)
        else:
            return array
        
    def band_arrays(self, mask_nodata=True):
        """Yields each band in the raster as an array, in order, along with its
        index.

        Parameters
        ----------
        mask_nodata : bool, optional
            if `True` NODATA values are masked in a returned `MaskedArray`.
            Else a simple `ndarray` is returned whith all values. True by
            default.

        Yields
        ------
        tuple (numpy.ndarray, int) or (numpy.ma.MaskedArray, int)
            Tuple with an array corresponding to each band, in order, and with
            the band index.
        """
        for i in range(self._count):
            yield (self.array_from_bands(i+1, mask_nodata=mask_nodata), i+1)

    def block_arrays(self, block_size=None, mask_nodata=True):
        """Yields each block in the raster as an array, in order, along with its
        xoffset and yoffset.

        Parameters
        ----------
        block_size : tuple of int (xsize, ysize), optional
            Size of blocks to yields. By default, "natural" block size is used.
        mask_nodata : bool, optional
            if `True` NODATA values are masked in a returned `MaskedArray`.
            Else a simple `ndarray` is returned whith all values. True by
            default.

        Yields
        ------
        tuple (numpy.ndarray, int, int) or (numpy.ma.MaskedArray, int, int)
            Tuple with an array corresponding to each block, in order, and with
            the block xoffset and yoffset.
        """
        for block_win in self.block_windows(block_size=block_size):
            yield (self.array_from_bands(block_win=block_win,
                                         mask_nodata=mask_nodata),
                   block_win[0],
                   block_win[1])

    def remove_bands(self, *idxs, **kw):
        """Saves a new raster with the specified bands removed.

        Parameters
        ----------
        idxs : int
            One or more indices of the band(s) to remove (numbering starts at 1)
        out_filename : str
            Path to the output file. If omitted, then the raster is overwritten

        Returns
        -------
        `Raster` or None
            Output raster or None if the raster is overwritten
        """
        indices = list(idxs)

        # Split the N-bands image into N mono-band images (in temp folder)
        SplitImage = otb.Registry.CreateApplication("SplitImage")
        SplitImage.SetParameterString("in", self._filename)
        SplitImage.SetParameterString("out", os.path.join(gettempdir(),
                                                          'splitted.tif'))
        SplitImage.SetParameterOutputImagePixelType(
            "out",
            self._dtype.otb_dtype)
        SplitImage.ExecuteAndWriteOutput()

        # Out file
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else os.path.join(gettempdir(), 'bands_removed.tif')

        # Concatenate the mono-band images without the unwanted band
        list_path = [os.path.join(gettempdir(), 'splitted_{}.tif'.format(i))
                     for i in range(self._count)
                     if i + 1 not in indices]
        ConcatenateImages = otb.Registry.CreateApplication("ConcatenateImages")
        ConcatenateImages.SetParameterStringList("il", list_path)
        ConcatenateImages.SetParameterString("out", out_filename)
        ConcatenateImages.SetParameterOutputImagePixelType(
            "out",
            self._dtype.otb_dtype)
        ConcatenateImages.ExecuteAndWriteOutput()

        # Delete mono-band images in temp folder
        for i in range(self._count):
            os.remove(os.path.join(gettempdir(), 'splitted_{}.tif'.format(i)))

        # Overwrite if wanted else return the new Raster
        if not kw.get('out_filename'):
            shutil.copy(out_filename, self._filename)
            os.remove(out_filename)
        else:
            return Raster(out_filename)

    def rescale_bands(self, dstmin, dstmax, *idxs, **kw):
        """Rescales one or more bands in the raster.

        For each specified band, values are rescaled between given minimum and
        maximum values.

        Parameters
        ----------
        dstmin : float
            Minimum value in the new scale.
        dstmax : float
            Maximum value in the new scale.
        idxs : int
            One or more indices of the bands to rescale.
        out_filename : str
            path to the output file. If omitted, then the raster is overwritten

        Returns
        -------
        `Raster` or None
            Output raster or None if the raster is overwritten
        """
        # Create an empty file with same size and dtype of float64
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else os.path.join(gettempdir(), 'bands_rescaled.tif')
        meta = self.meta
        meta['dtype'] = RasterDataType(gdal_dtype=gdal.GDT_Float64)
        write_file(out_filename, overwrite=True, **meta)

        # For each band, compute rescale if asked, then save band in empty file
        #        for i in range(1, self._count+1):
        for block_array, xoffset, yoffset in \
                self.block_arrays(mask_nodata=True):
            for i in idxs:
                try:
                    array = block_array[:, :, i]
                except IndexError:
                    if i != 1:
                        raise IndexError(
                            "Index out of range for mono-band image")
                    array = block_array
                srcmin = array.min()
                srcmax = array.max()
                array = dstmin + \
                    ((dstmax - dstmin) / (srcmax - srcmin)) \
                    * (array - srcmin)
            write_file(out_filename, array, xoffset=xoffset, yoffset=yoffset)

        # Overwrite if wanted else return the new Raster
        if not kw.get('out_filename'):
            shutil.copy(out_filename, self._filename)
            os.remove(out_filename)
        else:
            return Raster(out_filename)

    def fusion(self, pan, **kw):
        """Sharpen the raster with its corresponding panchromatic image.

        This function is quite conservative about extent and dates. The
        Panchromatic image should have same extent than the raster, and
        date/time of the two images should be the same.

        Parameters
        ----------
        pan : `Raster`
            Panchromatic image to use for sharpening.
        out_filename : str
            Path to the output file. If omitted, then the raster is overwritten.

        Returns
        -------
        `Raster` or `None`
            Output raster or `None` if the raster is overwritten.
        """
        # Check extents
        assert self.has_same_extent(pan), \
            "Images have not the same extent: '{:f}' and '{:f}'".format(
                self,
                pan)

        # Check dates and times
        assert self._date_time == pan.date_time, \
            "Images have not been taken at same time: '{:f}' and '{:f}'".format(
                self,
                pan)

        # Out file
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else os.path.join(gettempdir(), 'pan_sharpened.tif')

        # Actual sharpening
        Pansharpening = otb.Registry.CreateApplication("BundleToPerfectSensor")
        Pansharpening.SetParameterString("inp", pan.filename)
        Pansharpening.SetParameterString("inxs", self._filename)
        Pansharpening.SetParameterString("out", out_filename)
        Pansharpening.ExecuteAndWriteOutput()

        # Overwrite if wanted else return the new Raster
        if not kw.get('out_filename'):
            shutil.copy(out_filename, self._filename)
            os.remove(out_filename)
        else:
            return Raster(out_filename)

    @fix_missing_proj
    def radiometric_indices(self, *indices, **kw):
        """Saves a raster of radiometric indices about the raster.

        Parameters
        ----------
        indices : str
            Radiometric indices to compute.
        blue_idx: int
            index of a blue band (numbering starts at 1).
        green_idx: int
            index of a green band.
        red_idx: int
            index of a red band.
        nir_idx: int
            index of a nir band.
        mir_idx: int
            index of a mir band.
        out_filename: str
            Path to the output file. If omitted, a default filename will be
            chosen.

        Returns
        -------
        `Raster`
            Output raster
        """
        # Out file
        out_filename = None
        if kw.get('out_filename'):
            out_filename = kw['out_filename']
        else:
            inames = [rindex.split(':') for rindex in indices]
            inames = [name.lower() for _, name in inames]
            out_filename = '{:b}_{}.tif'.format(self, '_'.join(inames))

        # Actual computation
        RadiometricIndices = otb.Registry.CreateApplication(
            "RadiometricIndices")
        RadiometricIndices.SetParameterString("in", self._filename)
        try:
            RadiometricIndices.SetParameterInt("channels.blue",
                                               kw['blue_idx'])
        except KeyError:
            pass
        try:
            RadiometricIndices.SetParameterInt("channels.green",
                                               kw['green_idx'])
        except KeyError:
            pass
        try:
            RadiometricIndices.SetParameterInt("channels.red",
                                               kw['red_idx'])
        except KeyError:
            pass
        try:
            RadiometricIndices.SetParameterInt("channels.nir",
                                               kw['nir_idx'])
        except KeyError:
            pass
        try:
            RadiometricIndices.SetParameterInt("channels.mir", kw['mir_idx'])
        except KeyError:
            pass
        RadiometricIndices.SetParameterStringList("list", list(indices))
        RadiometricIndices.SetParameterString("out", out_filename)
        RadiometricIndices.ExecuteAndWriteOutput()

        out_raster = Raster(out_filename)
        if self._date_time:
            out_raster.date_time = self._date_time
        return out_raster

    def ndvi(self, red_idx, nir_idx, **kw):
        """Saves the Normalized Difference Vegetation Index (NDVI) of the
        raster.

        Parameters
        ----------
        red_idx : int
            Index of a red band (numbering starts at 1).
        nir_idx : int
            Index of a near-infrared band.
        out_filename : str
            Path to the output file. If omitted, a default filename will be
            chosen.

        Returns
        -------
        `Raster`
            Output raster.
        """
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_ndvi.tif'.format(self)
        return self.radiometric_indices("Vegetation:NDVI",
                                        red_idx=red_idx,
                                        nir_idx=nir_idx,
                                        out_filename=out_filename)

    def ndwi(self, nir_idx, mir_idx, **kw):
        """Saves the Normalized Difference Water Index (NDWI) of the raster.

        Parameters
        ----------
        nir_idx : int
            Index of a near infrared band (numbering starts at 1).
        mir_idx : int
            Index of the middle infrared band.
        out_filename : str
            path to the output file. If ommited, a default filename will be
            chosen.

        Returns
        -------
        `Raster`
            Output raster.
        """
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_ndwi.tif'.format(self)
        return self.radiometric_indices("Water:NDWI",
                                        nir_idx=nir_idx,
                                        mir_idx=mir_idx,
                                        out_filename=out_filename)

    def mndwi(self, green_idx, mir_idx, **kw):
        """Saves the Modified Normalized Difference Water Index (MNDWI) of the
        raster.

        Parameters
        ----------
        green_idx : int
            Index of a green band (numbering starts at 1).
        mir_idx : int
            Index of the middle infrared band.
        out_filename : str
            Path to the output file. If ommited, a default filename will be
            chosen.

        Returns
        -------
        `Raster`
            Output raster.
        """
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_mndwi.tif'.format(self)
        return self.radiometric_indices("Water:MNDWI",
                                        green_idx=green_idx,
                                        mir_idx=mir_idx,
                                        out_filename=out_filename)

    ndsi = mndwi

    def append(self, *rasters):
        """Append the given rasters into the current one.

        Parameters
        ----------
        rasters: `Raster`
            One or more rasters to append after the current one.
        """
        raster_list = [self] + list(rasters)
        concatenate_rasters(raster_list)

    def apply_mask(self, mask_raster, mask_value=1, set_value=None, **kw):
        """Apply a mask to the raster: set all masked pixels to a given value.

        NODATA will be set to set_value afterward.

        Parameters
        ----------
        mask_raster : `Raster`
            Mask to be applied.
        mask_value : float
            Value of "masked" pixels in the mask (default: 1).
        set_value : float
            Value to set to the "masked" pixels in the raster. If omitted,
            maximum value of the data type is chosen.
        out_filename : str
            Path of the output file. If omitted, a default filename will be
            chosen.

        Returns
        -------
        `Raster`
            Output raster.
        """
        # Check for extent
        assert self.has_same_extent(mask_raster), \
            "Images have not the same extent: '{:f}' and '{:f}'".format(
                self,
                mask_raster)

        # Set value
        try:
            set_value = set_value \
                if set_value \
                else np.iinfo(self.dtype.numpy_dtype).max
        except ValueError:
            set_value = np.finfo(self.dtype.numpy_dtype).max

        # Out file
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else os.path.join(gettempdir(), 'masked.tif')

        # Actual mask application
        for i in range(self._count):  # For each band
            # Compute the mask for that band
            BandMath = otb.Registry.CreateApplication("BandMath")
            expr = "(im2b1 >= {}) ? {} : im1b{}".format(mask_value,
                                                        set_value, i+1)
            temp_name = os.path.join(gettempdir(),
                                     'mono_masked_{}.tif'.format(i))
            BandMath.SetParameterStringList(
                "il",
                [self._filename, mask_raster.filename])
            BandMath.SetParameterString("out", temp_name)
            BandMath.SetParameterString("exp", expr)
            BandMath.ExecuteAndWriteOutput()
        # Then concatenate all masked bands and indicate the nodata_value
        masked_bands = [Raster(os.path.join(gettempdir(),
                                            'mono_masked_{}.tif'.format(i)))
                        for i in range(self._count)]
        concatenate_rasters(*masked_bands, out_filename=out_filename)
        out_raster = Raster(out_filename)
        out_raster.nodata_value = set_value

        # Delete the temp files
        for i in range(self._count):
            os.remove(os.path.join(gettempdir(),
                                   'mono_masked_{}.tif'.format(i)))

        return out_raster

    def _lsms_smoothing(self, spatialr, ranger, thres=0.1, rangeramp=0,
                        maxiter=10, **kw):
        """First step in a Large-Scale Mean-Shift (LSMS) segmentation: smooths
        the raster.

        This is an adapted version of the Orfeo Toolbox `MeanShiftSmoothing`
        application. See
        http://www.orfeo-toolbox.org/CookBook/CookBooksu91.html#x122-5480005.5.2
        for more details.

        Parameters
        ----------
        spatialr : int
            Spatial radius of the window (in number of pixels).
        ranger : float
            Range radius defining the spectral window size (expressed in
            radiometry unit).
        thres : float
            Mean shift vector threshold.
        rangeramp : float
            range radius coefficient. This coefficient makes dependent the
            `ranger` of the colorimetry of the filtered pixel: .. math::

                              y = rangeramp * x + ranger
        maxiter : int
            maximum number of iterations in case of non-convergence of the
            algorithm.
        out_spatial_filename : str
            Path to the spatial image to be written.
        out_filename : str
            Path to the smoothed file to be written.

        Returns
        -------
        tuple of 2 `Raster`
            filtered and spatial raster.
        """
        # Out files
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_smooth.tif'.format(self)
        out_spatial_filename = kw['out_spatial_filename'] \
            if kw.get('out_spatial_filename') \
            else '{:b}_spatial.tif'.format(self)

        # Actual smoothing
        MeanShiftSmoothing = otb.Registry.CreateApplication(
            "MeanShiftSmoothing")
        MeanShiftSmoothing.SetParameterString("in", self._filename)
        MeanShiftSmoothing.SetParameterString("fout", out_filename)
        MeanShiftSmoothing.SetParameterString("foutpos", out_spatial_filename)
        MeanShiftSmoothing.SetParameterInt("spatialr", spatialr)
        MeanShiftSmoothing.SetParameterFloat("ranger", ranger)
        MeanShiftSmoothing.SetParameterFloat("thres", thres)
        MeanShiftSmoothing.SetParameterFloat("rangeramp", rangeramp)
        MeanShiftSmoothing.SetParameterInt("maxiter", maxiter)
        MeanShiftSmoothing.SetParameterInt("modesearch", 0)
        MeanShiftSmoothing.ExecuteAndWriteOutput()

        return Raster(out_filename), Raster(out_spatial_filename)

    def _lsms_segmentation(self, spatialr, ranger, spatial_raster,
                           block_size=None, **kw):
        """Second step in a LSMS segmentation: performs the actual object
        segmentation on the raster.

        Produce an image whose pixels are given a label number, based on their
        spectral and spatial proximity.

        This assumes that the `Raster` object is smoothed, for example as
        returned by the `lsms_smoothing` method.

        To consume less memory resources, the method tiles the raster and
        performs the segmentation on each tile.

        This is an adapted version of the Orfeo Toolbox `LSMSSegmentation`
        application where there is no option to set a `minsize` parameter to
        discard small objects. See
        http://www.orfeo-toolbox.org/CookBook/CookBooksu121.html#x156-8990005.9.3
        for more details

        Parameters
        ----------
        spatialr : int
            Spatial radius of the window
        ranger : float
            Range radius defining the spectral window size (expressed in
            radiometry unit)
        block_size : tuple of int (xsize, ysize)
            wanted size for the blocks. To save memory, the segmentation work on
            blocks instead of the whole raster. If None, use the natural block
            size of the raster.
        spatial_raster : `Raster`
            Spatial raster associated to this raster (for example, as returned
            by the `lsms_smoothing` method)
        out_filename : str
            Path to the segmented image to be written

        Returns
        -------
        `Raster`
            Labeled raster.
        """
        # Blocks size
        tilesizex, tilesizey = block_size if block_size else self.block_size

        # Out file
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_label.tif'.format(self)

        # Actual segmentation
        LSMSSegmentation = otb.Registry.CreateApplication("LSMSSegmentation")
        LSMSSegmentation.SetParameterString("tmpdir", gettempdir())
        LSMSSegmentation.SetParameterString("in", self._filename)
        LSMSSegmentation.SetParameterString("inpos",
                                            spatial_raster.filename)
        LSMSSegmentation.SetParameterString("out", out_filename)
        LSMSSegmentation.SetParameterFloat("ranger", ranger)
        LSMSSegmentation.SetParameterFloat("spatialr", spatialr)
        LSMSSegmentation.SetParameterInt("minsize", 0)
        LSMSSegmentation.SetParameterInt("tilesizex", tilesizex)
        LSMSSegmentation.SetParameterInt("tilesizey", tilesizey)
        LSMSSegmentation.ExecuteAndWriteOutput()

        return Raster(out_filename)

    @fix_missing_proj
    def _lsms_merging(self, object_minsize, smoothed_raster, block_size=None,
                      **kw):
        """Third (optional) step in a LSMS segmentation:  merge objects in the
        raster whose size in pixels is lower than a given threshold into the
        bigger enough adjacent object with closest radiometry.

        This assumes that the `Raster` object is a segmented and labeled
        image, for example as returned by the `lsms_segmentation` method.

        The closest bigger object into which the small one is merged is
        determined by using the smoothed image which was produced by the first
        step of smoothing.

        This is an adapted version of the Orfeo Toolbox
        `LSMSSmallRegionsMerging` application. See
        http://www.orfeo-toolbox.org/CookBook/CookBooksu122.html#x157-9060005.9.4
        for more details.

        Parameters
        ----------
        object_minsize : int
            Threshold defining the minimum size of an object.
        smoothed_raster : `Raster`
            Smoothed raster associated to this raster (for example, as returned
            by the `lsms_smoothing` method)
        block_size : tuple of int (xsize, ysize)
            Wanted size for the blocks. To save memory, the merging work on
            blocks instead of the whole raster. If None, use the natural block
            size of the raster.
        out_filename : str
            path to the merged segmented image to be written.  If omitted,
            the raster will be overwritten.

        Returns
        -------
        `Raster`
            Merged segmented raster.
        """
        # Blocks size
        tilesizex, tilesizey = block_size if block_size else self.block_size

        # Out file
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else os.path.join(gettempdir(), 'labels.tif')

        # Actual merging
        LSMSSmallRegionsMerging = otb.Registry.CreateApplication(
            "LSMSSmallRegionsMerging")
        LSMSSmallRegionsMerging.SetParameterString("in",
                                                   smoothed_raster.filename)
        LSMSSmallRegionsMerging.SetParameterString("inseg", self._filename)
        LSMSSmallRegionsMerging.SetParameterString("out", out_filename)
        LSMSSmallRegionsMerging.SetParameterInt("minsize", object_minsize)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizex", tilesizex)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizey", tilesizey)
        LSMSSmallRegionsMerging.ExecuteAndWriteOutput()

        # Overwrite if wanted else return the new Raster
        if not kw.get('out_filename'):
            shutil.copy(out_filename, self._filename)
            os.remove(out_filename)
        else:
            return Raster(out_filename)

    def _lsms_vectorization(self, orig_raster, block_size=None, **kw):
        """Fourth and Last (optional) step in a LSMS segmentation: vectorize a
        labeled segmented image, turn each object into a polygon. Each polygon
        will have some attribute data:

            * the label number as an attribute,
            * the object's mean for each band in the original image,
            * the object's standard deviation for each band in the original
              image,
            * number of pixels in the object.

        This assumes that the `Raster` object is a segmented and labeled
        image, for example as returned by the `lsms_segmentation` or the
        `lsms_merging` methods.

        To consume less memory resources, the method tiles the raster and
        performs the segmentation on each tile.

        Parameters
        ----------
        orig_raster : `Raster`
            Original raster from which the segmentation was computed
        block_size : tuple of int (xsize, ysize)
            Wanted size for the blocks. To save memory, the vectorization work
            on blocks instead of the whole raster. If None, use the natural
            block size of the raster.
        out_filename : str
            Path to the output vector file. If omitted, a default filename will
            be chosen.
        """
        # Blocks size
        tilesizex, tilesizey = block_size \
            if block_size \
            else orig_raster.block_size

        # Out file
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_label.shp'.format(self)

        # Actual vectorization
        LSMSVectorization = otb.Registry.CreateApplication(
            "LSMSVectorization")
        LSMSVectorization.SetParameterString("in", orig_raster.filename)
        LSMSVectorization.SetParameterString("inseg", self._filename)
        LSMSVectorization.SetParameterString("out", out_filename)
        LSMSVectorization.SetParameterInt("tilesizex", tilesizex)
        LSMSVectorization.SetParameterInt("tilesizey", tilesizey)
        LSMSVectorization.ExecuteAndWriteOutput()

    def lsms_segmentation(self,
                          spatialr,
                          ranger,
                          thres=0.1,
                          rangeramp=0,
                          maxiter=10,
                          object_minsize=None,
                          block_size=None,
                          **kw):
        """Performs a Large-Scale-Mean-Shift (LSMS) object segmentation on the
        raster.

        Produces an image whose pixel values are label numbers, one label per
        object.

        Optionally, if the `out_vector_filename` parameter is given, then also
        writes a shapefile where each polygon is an object with its label number
        as an attribute.

        Parameters
        ----------
        spatialr : int
            Spatial radius in pixels. The algorithm compute the segmentation
            with a floating window which browse the image. This parameter
            specify the size of the window.
        ranger : float
            spectral range radius (expressed in radiometry unit).  This says how
            objects are determined from homogeneous pixels.
        thres : float
            Mean shift vector threshold
        rangeramp : float
            range radius coefficient. This coefficient makes dependent the
            `ranger` of the colorimetry of the filtered pixel: .. math::

                              y = rangeramp * x + ranger
        maxiter : int
            Maximum number of iterations in case of non-convergence of the
            algorithm
        object_minsize : int
            Threshold defining the minimum size in pixel of an object. If given,
            objects smaller than this size will be merged into a bigger adjacent
            object.
        tilesizex : int
            Horizontal size of each tile. To save memory, the segmentation work
            on tiles instead of the whole image.  If None, use the natural tile
            size of the image.
        tilesizey : int
            Vertical size of each tile. If None, use the natural tile size of
            the image.

        Returns
        -------
        `Raster`
            Labeled raster
        """

        # Temp filenames
        tmpdir = gettempdir()
        out_smoothed_filename = os.path.join(
            tmpdir, '{:b}_smoothed.tif'.format(self))
        out_spatial_filename = os.path.join(
            tmpdir, '{:b}_spatial.tif'.format(self))
        out_label_filename = os.path.join(
            tmpdir, '{:b}_label.tif'.format(self))

        # Out files
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_label.tif'.format(self)
        out_vector_filename = kw['out_vector_filename'] \
            if kw.get('out_vector_filename') \
            else '{:b}_label.shp'.format(self)

        # First step: smoothing
        smoothed_raster, spatial_raster = self._lsms_smoothing(
            spatialr=spatialr,
            ranger=ranger,
            thres=thres,
            rangeramp=rangeramp,
            maxiter=maxiter,
            out_filename=out_smoothed_filename,
            out_spatial_filename=out_spatial_filename)

        # Second step: actual object segmentation
        label_raster = smoothed_raster._lsms_segmentation(
            spatialr=spatialr,
            ranger=ranger,
            spatial_raster=spatial_raster,
            block_size=block_size,
            out_filename=out_label_filename)

        # Optional third step: merge small objects (< minsize) into bigger ones
        if object_minsize:
            label_raster._lsms_merging(
                object_minsize=object_minsize,
                smoothed_raster=smoothed_raster,
                block_size=block_size,
                out_filename=out_filename)
        else:
            shutil.copy(out_label_filename, out_filename)

        # Optional fourth step: convert into vector
        if kw.get('out_vector_filename') \
                or (not kw.get('out_vector_filename')
                    and not kw.get('out_filename')):
            out_raster = Raster(out_filename)
            out_raster._lsms_vectorization(
                orig_raster=self,
                block_size=block_size,
                out_filename=out_vector_filename)

        # Remove temp files
        for filename in (out_smoothed_filename, out_spatial_filename,
                         out_label_filename):
            os.remove(filename)

        if kw.get('out_filename'):
            return Raster(out_filename)
        else:
            os.remove(out_filename)

    def label_stats(self,
                    stats=['mean', 'std', 'min', 'max', "per:20",
                           'per:40', 'median', 'per:60', 'per:80'],
                    **kw):
        """Compute statistics from a labeled image.

        The statistics calculated by default are: mean, standard deviation, min,
        max and the 20, 40, 50, 60, 80th percentiles. The output is an image at
        the given format that contains n_band * n_stat_features bands.

        Parameters
        ----------
        label_raster : `Raster`
            The labeled raster.
        stats : list of str
            List of statistics to compute. By default: mean, std, min, max,
            per:20, per:40, per:50, per:60, per:80.
        out_filename : str
            Path of the output image. If omitted, a default filename is chosen.
        """
        # Create an empty file with correct size and dtype float64
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_label_stats.tif'.format(self)
        meta = self.meta
        meta['count'] = len(stats) * self._count
        meta['dtype'] = RasterDataType(gdal_dtype=gdal.GDT_Float64)
        write_file(out_filename, overwrite=True, **meta)

        # Get array of labels from label file
        label_raster = kw['label_raster'] \
            if kw.get('label_raster') \
            else None
        label_array = label_raster.array_from_bands()
        # Get array of unique labels
        unique_labels_array = np.unique(label_array)

        # Compute label stats
        i = 1
        # For each band
        for band_array, _ in self.band_arrays(mask_nodata=True):
            for statname in stats:                              # For each stat
                astat = array_stat.ArrayStat(statname)
                for label in unique_labels_array:               # For each label
                    # Compute stat for label
                    # all pixels of current label
                    label_indices = np.where(label_array == label)
                    # Compute all stats of all the pixels of the label
                    band_array[label_indices] = astat.compute(
                        band_array[label_indices])
                # Write the new band
                write_file(out_filename, band_array, band_idx=i)
                i += 1

    def get_stats_tierce(self, 
                         zones,
                         pixel_size,
                         stats_list=["mean", "std", "range", "median"], 
                         bands=["bl","ve","ro","br","ir"],
                         **kw):
        """Calcul statistics of the labels from a label image and a given raster. The
        statistics calculated by default are : mean, standard deviation, min,
        max and the 20, 40, 50, 60, 80th percentiles. The output is a vector file.
                Parameters
        ----------
        zones : OGR file
            Zones on on which stats are calculated

        pixel_size : float
            Pixel size of the output stats raster (i.e. pixel size of the image on which segmentation has been generated)        
        
        stats_list : list of str
            List of the statistics to be calculated. By default, all
            the features are calculated,i.e. mean, std, min, max, median, range and
            percentile. Percentile value must be written as "percentile_20".
        
        bands : list of str
            Ordered list of abbreviations of image bands, for fields name of 
            image stats (maximum size 6 characters)
        
        out_filename : str
            Path of the output image. If omitted, a default filename is chosen.        

        Returns
        -------
        `Raster`
            stats cube raster.
        """

        # Output cube
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_stats.tif'.format(self)
        
        # TOTO : Stats naming convention. Problème avec percentile_20 => gestion particulière du nommage 
        conv_dict = {'mean':'mea_',
         'std':'std_',
         'range':'ran_',
         'median':'med_',
         'percentile':'per_',
         'min' : 'min',
         'max' : 'max',
        }
        
        # create OGR driver
        driver = ogr.GetDriverByName('ESRI Shapefile')        
        
        # Delete unused fields, i.e. keep only "label" field
        os.system("ogr2ogr -q " + os.path.join(os.getcwd(),"temp.shp") + " " + zones)
        # Delete original zones OGR file for fields management (delete)       
        file_base = os.path.splitext(os.path.split(zones)[1])[0]
        folder_base = os.path.split(zones)[0]
        files = os.listdir(folder_base)
        for f in files:
            if os.path.splitext(f)[0] == file_base:
                full_path = os.path.join(folder_base,f)
                if not os.path.isdir(full_path):                
                    os.remove(full_path)          
                    
        # Delete fields
        dataSource_tmp = driver.Open("temp.shp", 1)
        layer_tmp = dataSource_tmp.GetLayer()
        layerDefinition_tmp = layer_tmp.GetLayerDefn()        
        nb_fields = layerDefinition_tmp.GetFieldCount()
        for no_field in range(nb_fields):
            if layerDefinition_tmp.GetFieldDefn(no_field).GetName() != "label":
                os.system('ogrinfo -q temp.shp -sql "ALTER TABLE temp DROP COLUMN %s\"' % layerDefinition_tmp.GetFieldDefn(no_field).GetName())
      
        # copy temp.shp to zones OGR file
        os.system('ogr2ogr -q -f "ESRI Shapefile" ' + zones + " " + os.path.join(os.getcwd(),"temp.shp"))        
        
        # delete temp.shp
        file_base = os.path.splitext("temp.shp")[0]
        folder_base = os.getcwd()
        files = os.listdir(folder_base)
        for f in files:
            if os.path.splitext(f)[0] == file_base:
                full_path = os.path.join(folder_base,f)
                if not os.path.isdir(full_path):                
                    os.remove(full_path)   
        
        # Open vector file in read-only mode
        dataSource = driver.Open(zones, 1)
        layer = dataSource.GetLayer()        
        
        # List Wkt str of each feature for zonal stats
        list_feat = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            list_feat.append(geom.ExportToWkt())
        
        layer.ResetReading()
        
        # Iterate bands, generate zonal stats for each and rename stats names
        allstats = []
        # Number of fieds / stats for iterate
        nb_stats = (self.meta['count'] * len(stats_list) * len(list_feat)) + self.meta['count']
        for i in range(self.meta['count']):
            stats_polygons = zonal_stats(list_feat, self.array_from_bands(i+1),stats=stats_list, transform=self.meta['transform'])                    
            for stats in stats_polygons:
                # Save keys names (original stat names)
                list_stat=[]                 
                for stat in stats:
                    list_stat.append(stat)
                # rename stat names, ex. "std_ir" 
                for stat in list_stat:
                    if stat != '__fid__' and nb_stats <> 0:                        
                        if stat.split("_")[0] == "percentile":
                            stats[conv_dict[stat.split("_")[0]]+stat.split("_")[1]+bands[i]] = stats.pop(stat)
                        else:
                            stats[conv_dict[stat]+bands[i]] = stats.pop(stat)
                        nb_stats -= 1
                
                allstats.append(stats)
                
        # OGR features number
        cpt_feat = layer.GetFeatureCount()
        # Stats number
        cpt_stats = len(allstats)
        
        # OGR output stats creation
        for stats_feat in allstats:                           
                if cpt_stats%cpt_feat == 0:
                    # créer les champs à chaque changement de bande 
                    for stat_feat_key,stat_feat_val in stats_feat.items():                
                        #print stat_feat_key,stat_feat_val
                        if stat_feat_key != "__fid__":                        
                            idField = ogr.FieldDefn(stat_feat_key, ogr.OFTReal)
                            idField.SetPrecision(3)
                            idField.SetWidth(15)
                            layer.CreateField(idField)
                  
                cpt_stats -= 1
        
        layer.GetNextFeature()
        for stats_feat in allstats:
            feature = layer.GetFeature(stats_feat["__fid__"])
            for stat_feat_key,stat_feat_val in stats_feat.items():
                if stat_feat_key != "__fid__":
                    feature.SetField(stat_feat_key, stat_feat_val)
            layer.SetFeature(feature)            
            
        # Rasterize each field into a "cube" raster
        layerDefinition = layer.GetLayerDefn()        
        nb_fields = layerDefinition.GetFieldCount()
        
        # Calculate width and height 
        hg, bg, hd, bd = self.meta['gdal_extent']
        x_res = int((bd[0] - bg[0])/pixel_size)
        y_res = int((hg[1] - bg[1])/pixel_size)

        # Create a memory mono-band raster to rasterize into             
        target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, nb_fields - 1,gdal.GDT_Float32)
        #target_ds.SetGeoTransform(self.meta["transform"])
        target_ds.SetGeoTransform((bg[0], pixel_size, 0, hg[1], 0, -pixel_size))
        target_ds.SetProjection(self.meta['srs'].ExportToWkt())         
        
        # Loop over stats fields and rasterise all values
        for i in range(nb_fields-1):          
            # Get field name            
            field = layerDefinition.GetFieldDefn(i).GetName()    
            print field
            # Rasterize field values into memory mono-band raster
            gdal.RasterizeLayer(target_ds, [i+1], layer, None, None, burn_values=[0], options = ["ATTRIBUTE=%s" % field])
            # Write the memory mono-band raster as a new band of cube output raster
        
        # Export raster
        gdal.GetDriverByName('GTiff').CreateCopy(out_filename,target_ds)
        
        # Statistics calculation
        ds = gdal.Open(out_filename, gdal.GA_ReadOnly)
        os.system("gdalinfo -q -stats " + out_filename)        
        ds.GetMetadata()

        # free memory
        target_ds = None
        dataSource.Destroy()
        
    def modify_metadata(self,
                        description):
                            
        xmlfile = self.filename + ".aux.xml"
        xmldoc = minidom.parse(xmlfile)
        for node in xmldoc.getElementsByTagName('Metadata'):
            x = xmldoc.createElement("MDI")
            x.setAttribute('key',"BAND")
            text = xmldoc.createTextNode(description)
            x.appendChild(text)
            node.appendChild(x)
            
        ofile = open(xmlfile,'w')
        xmldoc.writexml(ofile)
        ofile.close()
    
    def create_netcdf_monodate(self, metadata, out_filename):
        """Creation of netCDF file from raster file.
                Parameters
        ----------
        metadata : Metadata file (.XML), mandatory
            Metadata of the image files (based on DIMAP 2.0 format)
        
        out_filename : str
            Path of the output image. If omitted, a default filename is chosen.        

        Returns
        -------
        `Raster`
            stats cube raster.
        """        
        out_filename = kw['out_filename'] \
            if kw.get('out_filename') \
            else '{:b}_stats.tif'.format(self)
        
        # Get Metadata
        read_metadata(metadata)
        
        # Get wavelengths bounds (outside of Raster attributes)
        xmldoc = minidom.parse(metadata)        
        listband = []
        itemband = xmldoc.getElementsByTagName('Band_Spectral_Range')
        for i in range(0,len(itemband),1):
        listband.append([float(itemband[i].childNodes[11].firstChild.data) * 1000,float(itemband[i].childNodes[13].firstChild.data) * 1000])
        
        # number of characters to use in fixed-length strings.
        NUMCHARS = 8 
        # wavelengths bounds type
        wavebounds = numpy.dtype([('indwl',numpy.uint16),('namewl','S1',NUMCHARS),('minwl',numpy.float32),('maxwl',numpy.float32)])
        # NetCDF Compound type for wavelengths bounds
        wavebounds_t = rootgrp.createCompoundType(wavebounds,'wavelenghts_data')         
        
        # Create of NetCDF File
        rootgrp = Dataset(out_filename, "w", format="NETCDF4")
        
        # local Variable
        empty=()
        
        # Dimensions
        rootgrp.createDimension("x", self.width)
        rootgrp.createDimension("y", self.height)
        rootgrp.createDimension("band", self.count)
        
        # Lambert 93 projection for NetCDF
        lambert = rootgrp.createVariable("Lambert_Conformal","l",())
        rootgrp.variables["Lambert_Conformal"].grid_mapping_name = "lambert_conformal_conic"
        standard_parallel = []
        standard_parallel.append(osr.SpatialReference(wkt=self.srs.ExportToWkt()).GetProjParm("standard_parallel_1"))
        standard_parallel.append(osr.SpatialReference(wkt=self.srs.ExportToWkt()).GetProjParm("standard_parallel_2"))
        setattr(lambert,"standard_parallel",standard_parallel)
        setattr(lambert,"longitude_of_central_meridian",osr.SpatialReference(wkt=self.srs.ExportToWkt()).GetProjParm("Central_Meridian"))
        setattr(lambert,"latitude_of_projection_origin",osr.SpatialReference(wkt=self.srs.ExportToWkt()).GetProjParm("Latitude_Of_Origin"))
        setattr(lambert,"false_easting",osr.SpatialReference(wkt=self.srs.ExportToWkt()).GetProjParm("False_Easting")/1000)
        setattr(lambert,"false_northing",osr.SpatialReference(wkt=self.srs.ExportToWkt()).GetProjParm("False_Northing")/1000)
        setattr(lambert,"_CoordinateTransformType","Projection")
        setattr(lambert,"_CoordinateSystems","ProjectionCoordinateSystem")
        setattr(lambert,"_CoordinateAxes","y x")
        
        # Creation des variables
        x = rootgrp.createVariable("x","f4",("x",))
        rootgrp.variables["x"].units = "km"
        rootgrp.variables["x"].long_name = "x coordinate of projection"
        rootgrp.variables["x"].standard_name = "projection_x_coordinate"
        rootgrp.variables["x"]._CoordinateAxisType = "GeoX"
        
        y = rootgrp.createVariable("y","f4",("y",))
        rootgrp.variables["y"].units = "km"
        rootgrp.variables["y"].long_name = "y coordinate of projection"
        rootgrp.variables["y"].standard_name = "projection_y_coordinate"
        rootgrp.variables["y"]._CoordinateAxisType = "GeoY"
        
        lat = rootgrp.createVariable("lat","f4",("x","y"))
        rootgrp.variables["lat"].units = "degrees_north"
        rootgrp.variables["lat"].long_name = "latitude coordinate"
        rootgrp.variables["lat"].standard_name = "latitude"
        rootgrp.variables["lat"]._CoordinateAxisType = "Lat"
        
        lon = rootgrp.createVariable("lon","f4",("x","y"))
        rootgrp.variables["lon"].units = "degrees_east"
        rootgrp.variables["lon"].long_name = "longitude"
        rootgrp.variables["lon"].standard_name = "longitude coordinate"
        rootgrp.variables["lon"]._CoordinateAxisType = "Lon"
        
        rootgrp.createVariable("LatLonCoordinateSystem","c",empty)
        rootgrp.variables["LatLonCoordinateSystem"]._CoordinateAxes = "time lat lon"
        
        rootgrp.createVariable("ProjectionCoordinateSystem","c",empty)
        rootgrp.variables["ProjectionCoordinateSystem"]._CoordinateAxes = "time y x"
        rootgrp.variables["ProjectionCoordinateSystem"]._CoordinateTransforms = "LambertConformalProjection"
        
        wavelength = rootgrp.createVariable("wavelength",wavebounds_t,"band")
        rootgrp.variables["wavelength"].units = "nm"
        rootgrp.variables["wavelength"].long_name = "wavelength"
        
        reflectance = rootgrp.createVariable("reflectance",rastertype,("band","y","x"), zlib=True, fill_value=float(self._nodata_value), least_significant_digit=1)
        rootgrp.variables["reflectance"].long_name = "surface reflectance"
        rootgrp.variables["reflectance"].coordinates = "lat lon"
        rootgrp.variables["reflectance"].grid_mapping = "Lambert_Conformal"
        rootgrp.variables["reflectance"]._CoordinateSystems = "ProjectionCoordinateSystem LatLonCoordinateSystem"
        rootgrp.variables["reflectance"].missing_value = float(self._nodata_value)
        
        # Creation des axes de coordonnées, des bandes et des longueurs d'ondes
        x0 = self.transform[0] / 1000
        y1 = self.transform[3] / 1000
        y0 = y1 - (self.transform[1] / 1000 * self.width) + 0.001
        x1 = x0 + (self.transform[1] / 1000 * self.height)
        
        x[:] = numpy.arange(x0,x1,round(self.transform[1],2) / 1000)
        y[:] = numpy.arange(y1,y0,-round(self.transform[1],2) / 1000)
        wavelength[:] = numpy.arange(1,self.count + 1,1)
        
        # Wavelenghts metadata
        wl_metadata = numpy.empty(self.count,wavebounds_t)
        for i in range(self.count):
            wl_metadata['indwl'][i] = i + 1
            wl_metadata['namewl'][i] = stringtoarr('Band_%1d' % (i),8)
            wl_metadata['minwl'][i] = float(listband[i][0])
            wl_metadata['maxwl'][i] = float(listband[i][1])
        
        wavelength[:] = wl_metadata
        
        # Remplissage de la donnée
        for i in range(self.count):    
            reflectance[i,:,:] = self.array_from_bands(i + 1)
        
        # Enregistrement des metadonnees
        setattr(x,"valid_min",x0)
        setattr(x,"valid_max",x1)
        setattr(y,"valid_min",y0)
        setattr(y,"valid_max",y1)
        
        ma = numpy.ma.masked_equal(reflectance, 0.0, copy=False)
        setattr(reflectance,"valid_max",float(ma.max()))
        setattr(reflectance,"valid_min",float(ma.min()))
            
        setattr(rootgrp,"Date",datetime.datetime.strftime(self._date_time,'%Y-%m-%d'))
        setattr(rootgrp,"Time",datetime.datetime.strftime(self._date_time,'%H:%M:%S'))
        setattr(rootgrp,"title","Pléiades image of Coteaux de Gascogne")
            
        rootgrp.close()
        
        
    def read_metadata(self, metadata):
        """Set metadata information to Raster object (Date, time, range of spectral bands, no data.
                Parameters
        ----------
        metadata : Metadata file (.XML)
            Metadata of the image files (based on DIMAP 2.0 format)
        """
        
        xmldoc = minidom.parse(metadata)
        itemdate = xmldoc.getElementsByTagName('IMAGING_DATE')
        itemtime = xmldoc.getElementsByTagName('IMAGING_TIME')
        date = itemdate[0].childNodes[0].nodeValue
        time = itemtime[0].childNodes[0].nodeValue
        self._date_time = datetime.datetime.strptime(date+'-'+time[0:len(time)-3],'%Y-%m-%d-%H:%M:%S')
                
        listband = []
        itemband = xmldoc.getElementsByTagName('Band_Spectral_Range')
        for i in range(0,len(itemband),1):
            listband.append([float(itemband[i].childNodes[11].firstChild.data) * 1000, float(itemband[i].childNodes[13].firstChild.data) * 1000])
            self._wavelengths = lisband
        
        itemsv = xmldoc.getElementsByTagName('Special_Value')
        for i in range(0,len(itemsv),1): 
            if itemsv[i].childNodes[1].firstChild.data == "NODATA":
                self._nodata_value = float(itemsv[i].childNodes[3].firstChild.data)
        
        itempix = xmldoc.getElementsByTagName('DATA_TYPE')
        dt = itempix[0].firstChild.data
        itemnb = xmldoc.getElementsByTagName('NBITS')
        nbits = itemnb[0].firstChild.data           
        if dt == 'INTEGER':
            itemsi = xmldoc.getElementsByTagName('SIGN')
            sign = itemsi[0].firstChild.data
            if sign == 'UNSIGNED':
                if nbits == '8':
                    dtype = 'uint8'
                elif nbits == '16':
                    dtype = 'uint16'                    
                elif nbits == '32':
                    dtype = 'uint32'                    
            elif sign == 'SIGNED':
                if nbits == '8':
                    dtype = 'int8'
                elif nbits == '16':
                    dtype = 'int16'                    
                elif nbits == '32':
                    dtype = 'int32'                 
        elif dt == 'FLOAT':
            if nbits == '32':
                dtype = 'float32'
            elif nbits == '64':
                dtype = 'float64'              
        self._dtype = RasterDataType(lstr_dtype='int16') 

        for i in range(0,len(itempix),1):
            if itempix[i].childNodes[1].firstChild.data == "NODATA":
                self._nodata_value = float(itemsv[i].childNodes[3].firstChild.data)