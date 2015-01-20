# -*- coding: utf-8 -*-

import unittest

from ymraster import Raster


class TestRaster(unittest.TestCase):

    def setUp(self):
        self.filename = 'tests/data/RGB.byte.tif'
        self.raster = Raster(self.filename, 'blue', 'green', 'red')

    def test_should_get_attr_values(self):
        self.assertEqual(self.raster.filename, self.filename)
        self.assertEqual(self.raster.width, 791)
        self.assertEqual(self.raster.height, 718)
        self.assertEqual(self.raster.number_bands, 3)
        self.assertRegexpMatches(str(self.raster.crs), r'"EPSG", *"7030"')
        self.assertEqual(self.raster.topleft_x, 101985.0)
        self.assertEqual(self.raster.topleft_y, 2.826915e+06)
        self.assertAlmostEqual(self.raster.pixel_width, 300.038, places=3)
        self.assertAlmostEqual(self.raster.pixel_height, -300.042, places=3)
        self.assertEqual(self.raster.idx_blue, 0)
        self.assertEqual(self.raster.idx_green, 1)
        self.assertEqual(self.raster.idx_red, 2)


def suite():
    suite = unittest.TestSuite()
    load_from = unittest.defaultTestLoader.loadTestsFromTestCase
    suite.addTests(load_from(TestRaster))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
