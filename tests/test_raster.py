# -*- coding: utf-8 -*-

import unittest
import doctest
import tempfile

from ymraster import write_file, concatenate_rasters, Raster
from ymraster.dtype import RasterDataType
from osgeo import ogr, osr
import numpy as np

import re
import os
import shutil
from datetime import datetime
import subprocess


def write_file_unique_value(filename,
                            width,
                            height,
                            number_bands,
                            dtype,
                            value):
    array = np.ones((height, width, number_bands),
                    dtype=dtype.numpy_dtype) * value \
        if number_bands > 1 \
        else np.ones((height, width), dtype=dtype.numpy_dtype) * value
    write_file(filename, drivername='GTiff', dtype=dtype, array=array)


def _check_image(tester,
                 filename,
                 driver,
                 width,
                 height,
                 number_bands,
                 dtype,
                 proj=None,
                 transform=None,
                 date_time=None,
                 min_=None,
                 max_=None,
                 mean=None,
                 stddev=None):
    info = subprocess.check_output(["gdalinfo", "-stats", "-proj4",
                                    filename]).decode('utf-8')
    tester.assertRegexpMatches(info, u'Driver: {}'.format(driver))
    tester.assertRegexpMatches(info, u'Size is {}, {}'.format(width, height))
    tester.assertEqual(len(re.findall('Band \d+ ', info)), number_bands)
    tester.assertRegexpMatches(info, u'Type={}'.format(dtype))
    if proj is not None:
        tester.assertRegexpMatches(info, re.escape(proj))
    if min_ is not None:
        actual_min = float(re.search('Minimum=(-*[0-9\.]+),', info).group(1))
        tester.assertGreaterEqual(actual_min, min_)
    if max_ is not None:
        actual_max = float(re.search('Minimum=(-*[0-9\.]+),', info).group(1))
        tester.assertLessEqual(actual_max, max_)
    if mean is not None:
        tester.assertRegexpMatches(info, u'Mean={:.3f}, '.format(mean))
    if stddev is not None:
        tester.assertRegexpMatches(info, u'StdDev={:.3f}'.format(stddev))


class TestArrayToRaster(unittest.TestCase):

    def testwrite_file_one_band_square_uint16_image(self):
        width = 200
        height = 200
        number_bands = 1
        dtype = RasterDataType(numpy_dtype=np.uint16)
        value = 65535
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        write_file_unique_value(out_file.name,
                                width,
                                height,
                                number_bands,
                                dtype,
                                value)
        _check_image(self,
                     out_file.name,
                     u'GTiff',
                     width,
                     height,
                     number_bands,
                     dtype.ustr_dtype,
                     min_=value,
                     max_=value,
                     mean=value,
                     stddev=0)

    def testwrite_file_three_band_square_uint16_image(self):
        width = 400
        height = 400
        number_bands = 3
        dtype = RasterDataType(numpy_dtype=np.uint16)
        value = 65535
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        write_file_unique_value(out_file.name,
                                width,
                                height,
                                number_bands,
                                dtype,
                                value)
        _check_image(self,
                     out_file.name,
                     u'GTiff',
                     width,
                     height,
                     number_bands,
                     dtype.ustr_dtype,
                     min_=value,
                     max_=value,
                     mean=value,
                     stddev=0)

    def testwrite_file_three_band_rectangle_uint16_image(self):
        width = 800
        height = 600
        number_bands = 3
        dtype = RasterDataType(numpy_dtype=np.uint16)
        value = 65535
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        write_file_unique_value(out_file.name,
                                width,
                                height,
                                number_bands,
                                dtype,
                                value)
        _check_image(self,
                     out_file.name,
                     u'GTiff',
                     width,
                     height,
                     number_bands,
                     dtype.ustr_dtype,
                     min_=value,
                     max_=value,
                     mean=value,
                     stddev=0)

    def testwrite_file_should_raise_value_error_if_value_out_of_range(self):
        width = 3
        height = 3
        number_bands = 1
        dtype = RasterDataType(numpy_dtype=np.uint16)
        value = 65536
        a = np.ones((height, width, number_bands),
                    dtype=dtype.numpy_dtype) * value
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(ValueError, write_file, out_file.name,
                          drivername='GTiff', dtype=dtype, array=a)

    def testwrite_file_should_raise_not_implemented_if_four_dimensional(self):
        width = 3
        height = 3
        number_bands = 3
        dtype = RasterDataType(numpy_dtype=np.uint16)
        value = 65535
        a = np.ones((height, width, number_bands, 1),
                    dtype=dtype.numpy_dtype) * value
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(ValueError, write_file, out_file.name,
                          drivername='GTiff', dtype=dtype, array=a)

    def tearDown(self):
        tmpdir = tempfile.gettempdir()
        tmpfilenames = [filename
                        for filename in os.listdir(tmpdir)
                        if filename.endswith('.tif.aux.xml')]
        for filename in tmpfilenames:
            os.remove(os.path.join(tmpdir, filename))


class TestRaster(unittest.TestCase):

    def test_raster_should_get_attr_values(self):
        filename = 'data/l8_20130425.tif'
        raster = Raster(filename)
        self.assertEqual(raster.filename, filename)
        self.assertEqual(raster.meta['driver'].GetDescription(), u'GTiff')
        self.assertEqual(raster.meta['width'], 66)
        self.assertEqual(raster.meta['height'], 56)
        self.assertEqual(raster.meta['count'], 7)
        self.assertEqual(raster.meta['dtype'].lstr_dtype, 'int16')
        self.assertEqual(raster.meta['block_size'], (66, 8))
        self.assertEqual(raster.meta['date_time'], datetime(2013, 04, 25))
        self.assertEqual(raster.meta['gdal_extent'],
                         ((936306.723651, 6461635.694121),
                          (936306.723651, 6459955.694121),
                          (938286.723651, 6461635.694121),
                          (938286.723651, 6459955.694121)))
        self.assertEqual(
            raster.meta['srs'].ExportToWkt(),
            'PROJCS["RGF93 / Lambert-93",GEOGCS["RGF93",'
            'DATUM["Reseau_Geodesique_Francais_1993",'
            'SPHEROID["GRS 1980",6378137,298.2572221010042,'
            'AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6171"]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
            'AUTHORITY["EPSG","4171"]],'
            'PROJECTION["Lambert_Conformal_Conic_2SP"],'
            'PARAMETER["standard_parallel_1",49],'
            'PARAMETER["standard_parallel_2",44],'
            'PARAMETER["latitude_of_origin",46.5],'
            'PARAMETER["central_meridian",3],'
            'PARAMETER["false_easting",700000],'
            'PARAMETER["false_northing",6600000],UNIT["metre",1,'
            'AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","2154"]]')
        self.assertEqual(raster.meta['transform'], (936306.723651,
                                                    30.0,
                                                    0.0,
                                                    6461635.694121,
                                                    0.0,
                                                    -30.0))

    def test_raster_should_raise_runtime_error_on_missing_file(self):
        filename = 'data/not_exists.tif'
        self.assertRaises(RuntimeError, Raster, filename)

    def test_raster_should_raise_runtime_error_on_wrong_type_file(self):
        filename = 'data/foo.txt'
        self.assertRaises(RuntimeError, Raster, filename)

    def test_raster_should_get_block_list(self):
        filename = 'data/RGB.byte.tif'
        raster = Raster(filename)
        blocks = raster.block_windows()
        self.assertEqual(blocks.next(), (0, 0, raster.meta['block_size'][0],
                                         raster.meta['block_size'][1]))
        last_block = None
        length = 1
        total_height = raster.block_size[1]
        while True:
            try:
                last_block = blocks.next()
                length += 1
                total_height += last_block[3]
            except StopIteration:
                break
        self.assertEqual(last_block, (0, 717, raster.meta['block_size'][0], 1))
        self.assertEqual(length, 240)
        self.assertEqual(total_height, raster.meta['height'])

    def test_raster_should_get_block_list_with_given_size(self):
        filename = 'data/RGB.byte.tif'
        raster = Raster(filename)
        xsize = 7
        ysize = 2
        lastx = raster.meta['width'] - xsize
        lasty = raster.meta['height'] - ysize
        number_blocks = \
            raster.meta['width'] / xsize * raster.meta['height'] / ysize
        blocks = raster.block_windows(block_size=(xsize, ysize))
        self.assertEqual(blocks.next(), (0, 0, xsize, ysize))
        last_block = None
        length = 1
        while True:
            try:
                last_block = blocks.next()
                length += 1
            except StopIteration:
                break
        self.assertEqual(last_block, (lastx, lasty, xsize, ysize))
        self.assertEqual(length, number_blocks)

    def test_raster_should_get_array(self):
        filename = 'data/RGB.byte.tif'
        raster = Raster(filename)
        array = raster.array_from_bands()
        self.assertEqual(array.ndim, 3)
        self.assertEqual(array.shape, (718, 791, 3))
        self.assertEqual(array.dtype, 'UInt8')

    def test_raster_should_get_array_one_band(self):
        filename = 'data/RGB.byte.tif'
        raster = Raster(filename)
        array = raster.array_from_bands(1)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.shape, (718, 791))
        self.assertEqual(array.dtype, 'UInt8')

    def test_raster_should_get_array_block(self):
        filename = 'data/RGB.byte.tif'
        raster = Raster(filename)
        array = raster.array_from_bands(block_win=(256, 256, 300, 300))
        self.assertEqual(array.ndim, 3)
        self.assertEqual(array.shape, (300, 300, 3))
        self.assertEqual(array.dtype, 'UInt8')

    def test_raster_should_get_array_block_one_band(self):
        filename = 'data/RGB.byte.tif'
        raster = Raster(filename)
        array = raster.array_from_bands(2, block_win=(256, 256, 128, 128))
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.shape, (128, 128))
        self.assertEqual(array.dtype, 'UInt8')

    def test_raster_should_set_projection(self):
        filename = 'data/RGB_unproj.byte.tif'
        tmp_file = tempfile.NamedTemporaryFile(suffix='.tif')
        shutil.copyfile(filename, tmp_file.name)
        raster = Raster(tmp_file.name)
        self.assertIsNone(raster.srs)
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        raster.srs = sr
        _check_image(self,
                     tmp_file.name,
                     u'GTiff',
                     791,
                     718,
                     3,
                     'Byte',
                     proj=sr.ExportToProj4())
        self.assertTrue(raster.meta['srs'].IsSame(sr))

    def test_raster_should_set_date(self):
        filename = 'data/RGB_unproj.byte.tif'
        tmp_file = tempfile.NamedTemporaryFile(suffix='.tif')
        shutil.copyfile(filename, tmp_file.name)
        raster = Raster(tmp_file.name)
        self.assertIsNone(raster.meta['date_time'])
        dt = datetime(2014, 01, 01)
        raster.date_time = dt
        _check_image(self,
                     tmp_file.name,
                     u'GTiff',
                     791,
                     718,
                     3,
                     'Byte',
                     date_time=dt.strftime('%Y:%m:%d %H:%M:%S'))
        self.assertEqual(raster.meta['date_time'], dt)


class TestConcatenateImages(unittest.TestCase):

    def setUp(self):
        self.folder = 'data'

    def test_concatenate_should_work_when_same_size_same_proj(self):
        rasters = [Raster(os.path.join(self.folder, filename))
                   for filename in os.listdir(self.folder)
                   if filename.startswith('l8_')
                   and filename.endswith('.tif')]
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        concatenate_rasters(*rasters, out_filename=out_file.name)
        _check_image(tester=self,
                     filename=out_file.name,
                     driver=u'GTiff',
                     width=rasters[0].meta['width'],
                     height=rasters[0].meta['height'],
                     number_bands=rasters[0].meta['count'] * 8,
                     dtype=rasters[0].meta['dtype'].ustr_dtype,
                     proj=rasters[0].meta['srs'].ExportToProj4())

    def test_concatenate_should_raise_assertion_error_if_not_same_extent(self):
        rasters = [Raster(os.path.join(self.folder, 'shade.tif')),
                   Raster(os.path.join(self.folder, 'shade_crop.tif'))]
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(AssertionError, concatenate_rasters, *rasters,
                          outfilename=out_file.name)

    def test_concatenate_should_raise_assertion_error_if_not_same_proj(self):
        rasters = [Raster(os.path.join(self.folder, 'RGB.byte.tif')),
                   Raster(os.path.join(self.folder, 'RGB_unproj.byte.tif'))]
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(AssertionError, concatenate_rasters, *rasters,
                          out_filename=out_file.name)


class TestFusion(unittest.TestCase):

    def setUp(self):
        self.ms = Raster('data/Spot6_MS_31072013.tif')

    def test_fusion_should_work_if_same_date_same_extent(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        pan = Raster('data/Spot6_Pan_31072013.tif')
        self.ms.fusion(pan, out_filename=out_file.name)
        _check_image(tester=self,
                     filename=out_file.name,
                     driver=u'GTiff',
                     width=pan.meta['width'],
                     height=pan.meta['height'],
                     number_bands=self.ms.meta['count'],
                     dtype='Float32',
                     proj=self.ms.meta['srs'].ExportToProj4())

    def test_fusion_should_raise_assertion_error_if_not_same_extent(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        pan = Raster('data/Spot6_Pan_31072013_unproj.tif')
        self.assertRaises(AssertionError, self.ms.fusion, pan,
                          out_filename=out_file.name)

    def tearDown(self):
        tmpdir = tempfile.gettempdir()
        tmpfilenames = [filename
                        for filename in os.listdir(tmpdir)
                        if filename.endswith('.tif.aux.xml')]
        for filename in tmpfilenames:
            os.remove(os.path.join(tmpdir, filename))


class TestOtbFunctions(unittest.TestCase):

    def setUp(self):
        self.raster = Raster('data/l8_20130714.tif')

    def test_should_remove_band(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        result = self.raster.remove_bands(6, out_filename=out_file.name)
        self.assertEqual(result.meta['count'], self.raster.meta['count'] - 1)
        _check_image(tester=self,
                     filename=out_file.name,
                     driver=u'GTiff',
                     width=self.raster.meta['width'],
                     height=self.raster.meta['height'],
                     number_bands=self.raster.meta['count'] - 1,
                     dtype=self.raster.meta['dtype'].ustr_dtype,
                     proj=self.raster.meta['srs'].ExportToProj4())

    def test_should_compute_ndvi(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.raster.ndvi(red_idx=4, nir_idx=5, out_filename=out_file.name)
        _check_image(tester=self,
                     filename=out_file.name,
                     driver=u'GTiff',
                     width=self.raster.meta['width'],
                     height=self.raster.meta['height'],
                     number_bands=1,
                     dtype='Float32',
                     date_time=self.raster.meta['date_time'],
                     proj=self.raster.meta['srs'].ExportToProj4())

    def test_should_compute_ndwi(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.raster.ndwi(nir_idx=4, mir_idx=5, out_filename=out_file.name)
        _check_image(tester=self,
                     filename=out_file.name,
                     driver=u'GTiff',
                     width=self.raster.meta['width'],
                     height=self.raster.meta['height'],
                     number_bands=1,
                     dtype='Float32',
                     date_time=self.raster.meta['date_time'],
                     proj=self.raster.meta['srs'].ExportToProj4())

    def test_should_compute_mndwi(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.raster.mndwi(green_idx=4, mir_idx=5, out_filename=out_file.name)
        _check_image(tester=self,
                     filename=out_file.name,
                     driver=u'GTiff',
                     width=self.raster.meta['width'],
                     height=self.raster.meta['height'],
                     number_bands=1,
                     dtype='Float32',
                     date_time=self.raster.meta['date_time'],
                     proj=self.raster.meta['srs'].ExportToProj4())

    def test_lsms_segmentation_should_compute_segmented_image(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        out_vector_filename = os.path.join(
            tempfile.gettempdir(), 'labels.shp')
        out_raster = self.raster.lsms_segmentation(
            spatialr=5,
            ranger=15,
            thres=0.1,
            rangeramp=0,
            maxiter=5,
            object_minsize=10,
            out_vector_filename=out_vector_filename,
            out_filename=out_file.name)

        # Output raster should have same size, same proj, 1 band
        _check_image(tester=self,
                     filename=out_file.name,
                     driver=u'GTiff',
                     width=self.raster.meta['width'],
                     height=self.raster.meta['height'],
                     number_bands=1,
                     dtype='Float32',
                     proj=self.raster.meta['srs'].ExportToProj4())

        # Output vector should have same number of polygon than label
        array = out_raster.array_from_bands()
        number_labels = len(np.unique(array))
        ds = ogr.Open(out_vector_filename)
        layer = ds.GetLayer(0)
        self.assertEqual(layer.GetFeatureCount(), number_labels)

    def tearDown(self):
        tmpdir = tempfile.gettempdir()
        tmpfilenames = [filename
                        for filename in os.listdir(tmpdir)
                        if filename.endswith('.tif.aux.xml')
                        or filename.startswith('segmented_merged.')]
        for filename in tmpfilenames:
            os.remove(os.path.join(tmpdir, filename))


def unit_suite():
    suite = unittest.TestSuite()
    load_from = unittest.defaultTestLoader.loadTestsFromTestCase
    suite.addTests(load_from(TestRaster))
    return suite


def doc_suite():
    return doctest.DocTestSuite()


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(unit_suite())
    unittest.TextTestRunner(verbosity=2).run(doc_suite())
