# -*- coding: utf-8 -*-

try:
    import otbApplication as otb
except ImportError as e:
    raise ImportError(
        str(e)
        + "\n\nPlease install Orfeo Toolbox if it isn't installed yet.\n\n"
        "Also, add the otbApplication module path "
        "(usually something like '/usr/lib/otb/python') "
        "to the environment variable PYTHONPATH.\n")
try:
    from osgeo import gdal
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install GDAL.")
try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install NumPy.")


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

    data_type_match = {'uint8': otb.ImagePixelType_uint8,
                       'uint16': otb.ImagePixelType_uint16,
                       'uint32': otb.ImagePixelType_uint32,
                       'int16': otb.ImagePixelType_int16,
                       'int32': otb.ImagePixelType_int32,
                       'float32': otb.ImagePixelType_float,
                       'float64': otb.ImagePixelType_double}


class UStrDataType(DataType):
    """Represent a data type given in upper string format (eg. 'Int16', 'Int32',
    'Float32', etc.)"""

    data_type_match = {'UInt8': otb.ImagePixelType_uint8,
                       'UInt16': otb.ImagePixelType_uint16,
                       'UInt32': otb.ImagePixelType_uint32,
                       'Int16': otb.ImagePixelType_int16,
                       'Int32': otb.ImagePixelType_int32,
                       'Float32': otb.ImagePixelType_float,
                       'Float64': otb.ImagePixelType_double}


class NumpyDataType(DataType):
    """Represent a data type for Numpy (eg. np.int16, np.int32, np.float32,
    etc.)"""

    data_type_match = {np.uint8: otb.ImagePixelType_uint8,
                       np.uint16: otb.ImagePixelType_uint16,
                       np.uint32: otb.ImagePixelType_uint32,
                       np.int16: otb.ImagePixelType_int16,
                       np.int32: otb.ImagePixelType_int32,
                       np.float32: otb.ImagePixelType_float,
                       np.float64: otb.ImagePixelType_double}


class GdalDataType(DataType):
    """Represent a data type for gdal (eg. gdal.GDT_Int16, gdal.GDT_Iint32,
    gdal.GDT_Float32, etc.)"""

    data_type_match = {gdal.GDT_Byte: otb.ImagePixelType_uint8,
                       gdal.GDT_UInt16: otb.ImagePixelType_uint16,
                       gdal.GDT_UInt32: otb.ImagePixelType_uint32,
                       gdal.GDT_Int16: otb.ImagePixelType_int16,
                       gdal.GDT_Int32: otb.ImagePixelType_int32,
                       gdal.GDT_Float32: otb.ImagePixelType_float,
                       gdal.GDT_Float64: otb.ImagePixelType_double}


class OtbDataType(DataType):
    """Represent a data type for orfeo-toolbox
    (eg. otb.ImagePixelType_int16)"""

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
