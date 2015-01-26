# -*- coding: utf-8 -*-

import unittest
import mock
import tempfile

from ymraster import _save_array, concatenate_images, Raster
import numpy as np

import os
import re
import subprocess


def _check_output_image(tester, filename, driver, width, height, number_bands,
                        dtype, min_=None, max_=None, mean=None, stddev=None):
    dtype_str_match = {np.uint16: 'UInt16',
                       np.int16: 'Int16'}
    info = subprocess.check_output(["gdalinfo", "-stats",
                                    filename]).decode('utf-8')
    tester.assertRegexpMatches(info, u'Driver: {}'.format(driver))
    tester.assertRegexpMatches(info, u'Size is {}, {}'.format(width, height))
    tester.assertEqual(len(re.findall('Band \d+ ', info)), number_bands)
    tester.assertRegexpMatches(info, u'Type={}'.format(dtype_str_match[dtype]))
    if min_ is not None:
        tester.assertRegexpMatches(info, u'Minimum={:.3f}, '.format(min_))
    if max_ is not None:
        tester.assertRegexpMatches(info, u'Maximum={:.3f}, '.format(max_))
    if mean is not None:
        tester.assertRegexpMatches(info, u'Mean={:.3f}, '.format(mean))
    if stddev is not None:
        tester.assertRegexpMatches(info, u'StdDev={:.3f}'.format(stddev))


class TestArrayToRaster(unittest.TestCase):

    def _execute_save_array(self):
        if self.number_bands == 1:
            a = np.ones((self.height, self.width),
                        dtype=self.dtype) * self.value
        else:
            a = np.ones((self.height, self.width, self.number_bands),
                        dtype=self.dtype) * self.value
        meta = {'driver': self.driver,
                'width': self.width,
                'height': self.height,
                'count': self.number_bands,
                'dtype': self.dtype}
        _save_array(a, self.filename, meta)

    def setUp(self):
        self.driver = u'GTiff'

    def test_should_write_one_band_square_uint16_image(self):
        self.width = 200
        self.height = 200
        self.number_bands = 1
        self.dtype = np.uint16
        self.value = 65535
        f = tempfile.NamedTemporaryFile(suffix='.tif')
        self.filename = f.name
        self._execute_save_array()
        _check_output_image(self, self.filename, self.driver, self.width,
                            self.height, self.number_bands, self.dtype,
                            self.value, self.value, self.value, 0)

    def test_should_write_three_band_square_uint16_image(self):
        self.width = 400
        self.height = 400
        self.number_bands = 3
        self.dtype = np.uint16
        self.value = 65535
        f = tempfile.NamedTemporaryFile(suffix='.tif')
        self.filename = f.name
        self._execute_save_array()
        _check_output_image(self, self.filename, self.driver, self.width,
                            self.height, self.number_bands, self.dtype,
                            self.value, self.value, self.value, 0)

    def test_should_write_three_band_rectangle_uint16_image(self):
        self.width = 800
        self.height = 600
        self.number_bands = 3
        self.dtype = np.uint16
        self.value = 65535
        f = tempfile.NamedTemporaryFile(suffix='.tif')
        self.filename = f.name
        self._execute_save_array()
        _check_output_image(self, self.filename, self.driver, self.width,
                            self.height, self.number_bands, self.dtype,
                            self.value, self.value, self.value, 0)

    def test_should_raise_value_error_if_value_out_of_range(self):
        width = 3
        height = 3
        number_bands = 1
        dtype = np.uint16
        value = 65536
        a = np.ones((height, width, number_bands), dtype=dtype) * value
        meta = {'driver': self.driver,
                'width': width,
                'height': height,
                'count': number_bands,
                'dtype': dtype}
        f = tempfile.NamedTemporaryFile(suffix='.tif')
        name = f.name
        self.assertRaises(ValueError, _save_array, a, name, meta)

    def test_should_raise_not_implemented_if_array_four_dimensional(self):
        width = 3
        height = 3
        number_bands = 3
        dtype = np.uint16
        value = 65535
        a = np.ones((height, width, number_bands, 1), dtype=dtype) * value
        meta = {'driver': self.driver,
                'width': width,
                'height': height,
                'count': number_bands,
                'dtype': dtype}
        f = tempfile.NamedTemporaryFile(suffix='.tif')
        name = f.name
        self.assertRaises(NotImplementedError, _save_array, a, name, meta)


class TestConcatenateImages(unittest.TestCase):

    def setUp(self):
        self.folder = 'tests/data'
        self.driver = u'GTiff'

    def test_should_concatenate_same_type_images(self):
        width = 66
        height = 56
        number_bands = 35
        dtype = np.int16
        rasters = [Raster(os.path.join(self.folder, filename))
                   for filename in os.listdir(self.folder)
                   if filename.startswith('l8_')
                   and filename.endswith('.tif')]
        f = tempfile.NamedTemporaryFile(suffix='.tif')
        filename = f.name
        concatenate_images(rasters, filename)
        _check_output_image(self, filename, self.driver, width, height,
                            number_bands, dtype)

    def test_should_raise_assertion_error_if_not_same_size(self):
        rasters = [Raster(os.path.join(self.folder, 'RGB.byte.tif')),
                   Raster(os.path.join(self.folder, 'float.tif'))]
        f = tempfile.NamedTemporaryFile(suffix='.tif')
        filename = f.name
        self.assertRaises(AssertionError, concatenate_images, rasters, filename)


class TestRealRaster(unittest.TestCase):

    def setUp(self):
        self.filename = 'tests/data/RGB.byte.tif'
        self.raster = Raster(self.filename)

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

    def test_should_get_array(self):
        array = self.raster.array()
        self.assertEqual(array.ndim, 3)
        self.assertEqual(array.shape, (718, 791, 3))
        self.assertEqual(array.dtype, np.uint8)

    def test_should_raise_io_error_on_missing_file(self):
        filename = 'tests/data/not_exists.tif'
        self.assertRaises(IOError, Raster, filename)


class TestRasterArrayFunctions(unittest.TestCase):

    def setUp(self):
        self.filename = 'tests/data/RGB.byte.tif'
        self.raster = Raster(self.filename)
        self.raster.array = mock.Mock(
            return_value=np.arange(1, 37).reshape(3, 3, 4))

    def test_should_compute_correct_ndvi(self):
        self.assertEqual(self.raster.filename, self.filename)


class TestOtbFunctions(unittest.TestCase):

    def _check_normalized_index(self):
        self.assertEqual(self.result.meta['count'], 1)
        info = subprocess.check_output(["gdalinfo", "-stats",
                                        self.f.name]).decode('utf-8')
        self.assertRegexpMatches(info, u'Driver: GTif')
        self.assertRegexpMatches(info, u'Size is 66, 56')
        self.assertRegexpMatches(info, u'Type=Float32')
        match = re.search('Minimum=(-*[0-9\.]+), Maximum=(-*[0-9\.]+),', info)
        min = float(match.group(1))
        max = float(match.group(2))
        self.assertGreaterEqual(min, -1)
        self.assertLessEqual(max, 1)

    def setUp(self):
        self.filename = 'tests/data/l8_20130714.tif'
        self.raster = Raster(self.filename)
        self.f = tempfile.NamedTemporaryFile(suffix='.tif')

    def test_should_remove_band(self):
        self.result = self.raster.remove_band(6, self.f.name)
        self.assertEqual(self.result.meta['count'],
                         self.raster.meta['count'] - 1)
        info = subprocess.check_output(["gdalinfo", "-stats",
                                        self.f.name]).decode('utf-8')
        raster_info = subprocess.check_output(["gdalinfo", "-stats",
                                               self.filename]).decode('utf-8')
        raster_driver = re.search('(Driver: \w+)', raster_info).group(1)
        raster_size = re.search('(Size is \d+, \d+)', raster_info).group(1)
        raster_dtype = re.search('(Type=\w+),', raster_info).group(1)
        self.assertRegexpMatches(info, raster_driver)
        self.assertRegexpMatches(info, raster_size)
        self.assertRegexpMatches(info, raster_dtype)

    def test_should_compute_ndvi(self):
        self.result = self.raster.ndvi(self.f.name, 4, 5)
        self._check_normalized_index()

    def test_should_compute_ndwi(self):
        self.result = self.raster.ndwi(self.f.name, 5, 6)
        self._check_normalized_index()

    def test_should_compute_mndwi(self):
        self.result = self.raster.ndwi(self.f.name, 3, 6)
        self._check_normalized_index()


def suite():
    suite = unittest.TestSuite()
    load_from = unittest.defaultTestLoader.loadTestsFromTestCase
    suite.addTests(load_from(TestRealRaster))
    suite.addTests(load_from(TestRasterArrayFunctions))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
