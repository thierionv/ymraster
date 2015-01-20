# -*- coding: utf-8 -*-

import unittest

from ymraster import Raster

class TestRaster(unittest.TestCase):

    def setUp(self):
        filename = 'tests/data/RGB.byte.tif'
        self.raster = Raster(filename, 'blue', 'green', 'red')

    def test_should_get_attr_values(self):
        self.assertEqual(self.raster.width, 791)

def suite():
    suite = unittest.TestSuite()
    load_from = unittest.defaultTestLoader.loadTestsFromTestCase
    suite.addTests(load_from(TestRaster))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
