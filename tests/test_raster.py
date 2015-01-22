# -*- coding: utf-8 -*-

import unittest
import mock
import tempfile

from ymraster import _save_array, Raster
import numpy as np

import subprocess


class TestArrayToRaster(unittest.TestCase):

    def _execute_save_array(self):
        if self.n_bands == 1:
            a = np.ones((self.height, self.width),
                        dtype=self.dtype) * self.value
        else:
            a = np.ones((self.height, self.width, self.n_bands),
                        dtype=self.dtype) * self.value
        meta = {'driver': self.driver,
                'width': self.width,
                'height': self.height,
                'count': self.n_bands,
                'dtype': self.dtype}
        _save_array(a, self.name, meta)

    def _check_output_image(self):
        self.assertRegexpMatches(self.info, u'Driver: {}'.format(self.driver))
        self.assertRegexpMatches(self.info, u'Size is {}, {}'.format(
            self.width,
            self.height))
        self.assertRegexpMatches(self.info, u'Type=UInt16')
        self.assertRegexpMatches(self.info, u'Minimum={0:.3f}, '
                                 'Maximum={0:.3f}, '
                                 'Mean={0:.3f}, '
                                 'StdDev={1:.3f}'.format(self.value, 0))

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile()
        self.name = self.f.name
        self.driver = u'GTiff'

    def test_should_write_one_band_square_uint16_image(self):
        self.width = 200
        self.height = 200
        self.n_bands = 1
        self.dtype = np.uint16
        self.value = 65535
        self._execute_save_array()
        self.info = subprocess.check_output(["gdalinfo", "-stats",
                                             self.name]).decode('utf-8')
        self._check_output_image()

    def test_should_write_three_band_square_uint16_image(self):
        self.width = 400
        self.height = 400
        self.n_bands = 3
        self.dtype = np.uint16
        self.value = 65535
        self._execute_save_array()
        self.info = subprocess.check_output(["gdalinfo", "-stats",
                                             self.name]).decode('utf-8')
        self._check_output_image()

    def test_should_write_three_band_rectangle_uint16_image(self):
        self.width = 800
        self.height = 600
        self.n_bands = 3
        self.dtype = np.uint16
        self.value = 65535
        self._execute_save_array()
        self.info = subprocess.check_output(["gdalinfo", "-stats",
                                             self.name]).decode('utf-8')
        self._check_output_image()

    def test_should_raise_value_error_if_value_out_of_range(self):
        self.width = 3
        self.height = 3
        self.n_bands = 1
        self.dtype = np.uint16
        self.value = 65536
        a = np.ones((self.height, self.width, self.n_bands),
                    dtype=self.dtype) * self.value
        meta = {'driver': self.driver,
                'width': self.width,
                'height': self.height,
                'count': self.n_bands,
                'dtype': self.dtype}
        self.assertRaises(ValueError, _save_array, a, self.name, meta)

    def test_should_raise_not_implemented_if_array_four_dimensional(self):
        self.width = 3
        self.height = 3
        self.n_bands = 3
        self.dtype = np.uint16
        self.value = 65535
        a = np.ones((self.height, self.width, self.n_bands, 1),
                    dtype=self.dtype) * self.value
        meta = {'driver': self.driver,
                'width': self.width,
                'height': self.height,
                'count': self.n_bands,
                'dtype': self.dtype}
        self.assertRaises(NotImplementedError, _save_array, a, self.name, meta)


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
