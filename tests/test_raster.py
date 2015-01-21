# -*- coding: utf-8 -*-

import unittest
import mock

from ymraster import Raster
import numpy as np


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
    suite.addTests(load_from(TestRasterIndices))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
