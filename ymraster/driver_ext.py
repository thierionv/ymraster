# -*- coding: utf-8 -*-

try:
    from osgeo import gdal
except ImportError as e:
    raise ImportError(
        str(e) + "\n\nPlease install GDAL.")

from collections import namedtuple


GenericDriverExt = namedtuple('GenericDriverExt',
                              ['extension', 'gdal_name', 'gdal_driver'])


class DriverExt(GenericDriverExt):
    """Class to map gdal.Driver instance with filename extensions"""

    drivername_map = {'.tif': 'GTiff',
                      '.h5': 'HDF5'}

    __slots__ = ()

    def __new__(cls, extension=None, gdal_driver=None):
        if extension:
            try:
                driver = gdal.GetDriverByName(cls.drivername_map[extension])
            except KeyError:
                raise NotImplementedError(
                    "No driver has been mapped to extension: '{}'".format(
                        extension))
            return super(DriverExt, cls).__new__(cls,
                                                 extension,
                                                 driver.ShortName,
                                                 driver)
        elif gdal_driver:
            extension_map = {v: k for k, v in cls.drivername_map.iteritems()}
            try:
                extension = extension_map[gdal_driver.ShortName]
            except KeyError:
                raise NotImplementedError(
                    "No extension has been mapped to driver: '{}'".format(
                        gdal_driver.ShortName))
            return super(DriverExt, cls).__new__(cls,
                                                 extension,
                                                 gdal_driver.ShortName,
                                                 gdal_driver)
