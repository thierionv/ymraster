# -*- coding: utf-8 -*-

import unittest
import mock
import tempfile

from ymraster import _save_array, Raster
import numpy as np

import subprocess


class TestArrayToRaster(unittest.TestCase):

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile()
        self.name = self.f.name

    def test_should_write_one_band_image(self):
        width = 3
        height = 3
        n_bands = 1
        dtype = np.uint16
        value = 65535
        a = np.ones((width, height), dtype=dtype) * value
        meta = {'driver': u'GTiff', 'width': width, 'height': height,
                'count': n_bands, 'dtype': dtype}
        _save_array(a, self.name, meta)
        stats = subprocess.check_output(["gdalinfo", "-stats",
                                         self.name]).decode('utf-8')
        self.assertRegexpMatches(stats, u'Driver: GTiff')
        self.assertRegexpMatches(stats, u'Size is {}, {}'.format(height, width))
        self.assertRegexpMatches(stats, u'Type=UInt16')
        self.assertRegexpMatches(stats, u'Minimum=65535.000, '
                                 'Maximum=65535.000, '
                                 'Mean=65535.000, '
                                 'StdDev=0.000')


class TestRealRaster(unittest.TestCase):

    def setUp(self):
        self.filename = 'tests/data/RGB.byte.tif'
        self.raster = Raster(self.filename, 'blue', 'green', 'red')

    def test_should_get_attr_values(self):
        self.assertEqual(self.raster.filename, self.filename)
        self.assertEqual(self.raster.meta['count'], 3)
        self.assertEqual(self.raster.meta['crs'],
                         {'init': u'epsg:32618'})
        self.assertEqual(self.raster.meta['dtype'], 'uint8')
        self.assertEqual(self.raster.meta['driver'], u'GTiff')
        self.assertEqual(self.raster.meta['transform'], (101985.0,
                                                         300.0379266750948,
                                                         0.0,
                                                         2826915.0,
                                                         0.0,
                                                         -300.041782729805))
        self.assertEqual(self.raster.meta['height'], 718)
        self.assertEqual(self.raster.meta['width'], 791)
        self.assertEqual(self.raster.meta['nodata'], 0.0)


class TestRasterIndices(unittest.TestCase):

    def setUp(self):
        self.filename = 'tests/data/RGB.byte.tif'
        self.raster = Raster(self.filename, 'blue', 'green', 'red')
        self.raster.array = mock.Mock(
            return_value=np.arange(1, 37).reshape(3, 3, 4))

    def test_should_compute_correct_ndvi(self):
        self.assertEqual(self.raster.filename, self.filename)


def suite():
    suite = unittest.TestSuite()
    load_from = unittest.defaultTestLoader.loadTestsFromTestCase
    suite.addTests(load_from(TestRealRaster))
    suite.addTests(load_from(TestRasterIndices))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
