# -*- coding: utf-8 -*-

import unittest
# import mock
import tempfile

from ymraster import _save_array, concatenate_images, Raster
from osgeo import gdal, ogr
import numpy as np

import os
import re
import subprocess


def _save_array_unique_value(filename,
                             width,
                             height,
                             number_bands,
                             dtype,
                             value):
    array = np.ones((height, width, number_bands), dtype=dtype) * value \
        if number_bands > 1 \
        else np.ones((height, width), dtype=dtype) * value
    meta = {'driver': u'GTiff',
            'width': width,
            'height': height,
            'count': number_bands,
            'dtype': dtype}
    _save_array(array, filename, meta)


def _check_output_image(tester,
                        filename,
                        driver,
                        width,
                        height,
                        number_bands,
                        dtype,
                        proj=None,
                        transform=None,
                        date=None,
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

    def test_save_array_one_band_square_uint16_image(self):
        width = 200
        height = 200
        number_bands = 1
        dtype = 'UInt16'
        value = 65535
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        _save_array_unique_value(out_file.name,
                                 width,
                                 height,
                                 number_bands,
                                 dtype,
                                 value)
        _check_output_image(self,
                            out_file.name,
                            u'GTiff',
                            width,
                            height,
                            number_bands,
                            dtype,
                            min_=value,
                            max_=value,
                            mean=value,
                            stddev=0)

    def test_save_array_three_band_square_uint16_image(self):
        width = 400
        height = 400
        number_bands = 3
        dtype = 'UInt16'
        value = 65535
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        _save_array_unique_value(out_file.name,
                                 width,
                                 height,
                                 number_bands,
                                 dtype,
                                 value)
        _check_output_image(self,
                            out_file.name,
                            u'GTiff',
                            width,
                            height,
                            number_bands,
                            dtype,
                            min_=value,
                            max_=value,
                            mean=value,
                            stddev=0)

    def test_save_array_three_band_rectangle_uint16_image(self):
        width = 800
        height = 600
        number_bands = 3
        dtype = 'UInt16'
        value = 65535
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        _save_array_unique_value(out_file.name,
                                 width,
                                 height,
                                 number_bands,
                                 dtype,
                                 value)
        _check_output_image(self,
                            out_file.name,
                            u'GTiff',
                            width,
                            height,
                            number_bands,
                            dtype,
                            min_=value,
                            max_=value,
                            mean=value,
                            stddev=0)

    def test_save_array_should_raise_value_error_if_value_out_of_range(self):
        width = 3
        height = 3
        number_bands = 1
        dtype = 'UInt16'
        value = 65536
        a = np.ones((height, width, number_bands), dtype=dtype) * value
        meta = {'driver': u'GTiff',
                'width': width,
                'height': height,
                'count': number_bands,
                'dtype': dtype}
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(ValueError, _save_array, a, out_file.name, meta)

    def test_save_array_should_raise_not_implemented_if_four_dimensional(self):
        width = 3
        height = 3
        number_bands = 3
        dtype = 'UInt16'
        value = 65535
        a = np.ones((height, width, number_bands, 1), dtype=dtype) * value
        meta = {'driver': u'GTiff',
                'width': width,
                'height': height,
                'count': number_bands,
                'dtype': dtype}
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(NotImplementedError, _save_array, a, out_file.name,
                          meta)

    def tearDown(self):
        tmpdir = tempfile.gettempdir()
        tmpfilenames = [filename
                        for filename in os.listdir(tmpdir)
                        if filename.endswith('.tif.aux.xml')]
        for filename in tmpfilenames:
            os.remove(os.path.join(tmpdir, filename))


class TestConcatenateImages(unittest.TestCase):

    def setUp(self):
        self.folder = 'tests/data'

    def test_concatenate_should_work_when_same_type_same_proj(self):
        rasters = [Raster(os.path.join(self.folder, filename))
                   for filename in os.listdir(self.folder)
                   if filename.startswith('l8_')
                   and filename.endswith('.tif')]
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        concatenate_images(rasters, out_file.name)
        _check_output_image(tester=self,
                            filename=out_file.name,
                            driver=u'GTiff',
                            width=rasters[0].meta['width'],
                            height=rasters[0].meta['height'],
                            number_bands=rasters[0].meta['count'] * 5,
                            dtype=rasters[0].meta['dtype'].ustr_dtype,
                            proj=rasters[0].meta['srs'].ExportToProj4())

    def test_concatenate_should_raise_assertion_error_if_not_same_size(self):
        rasters = [Raster(os.path.join(self.folder, 'RGB.byte.tif')),
                   Raster(os.path.join(self.folder, 'float.tif'))]
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(AssertionError, concatenate_images, rasters,
                          out_file.name)

    def test_concatenate_should_raise_assertion_error_if_not_same_proj(self):
        rasters = [Raster(os.path.join(self.folder, 'RGB.byte.tif')),
                   Raster(os.path.join(self.folder, 'RGB_unproj.byte.tif'))]
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(AssertionError, concatenate_images, rasters,
                          out_file.name)


class TestFusion(unittest.TestCase):

    def setUp(self):
        self.ms = Raster('tests/data/Spot6_MS_31072013.tif')
        self.pan = Raster('tests/data/Spot6_Pan_31072013.tif')

    def test_fusion_should_work_if_same_date_same_projection(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.ms.fusion(self.pan, out_file.name)
        _check_output_image(tester=self,
                            filename=out_file.name,
                            driver=u'GTiff',
                            width=self.pan.meta['width'],
                            height=self.pan.meta['height'],
                            number_bands=self.ms.meta['count'],
                            dtype='Float32',
                            proj=self.ms.meta['srs'].ExportToProj4())

    def test_fusion_should_raise_assertion_error_if_not_same_proj(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.assertRaises(AssertionError, self.ms.fusion, self.pan,
                          out_file.name)

    def tearDown(self):
        tmpdir = tempfile.gettempdir()
        tmpfilenames = [filename
                        for filename in os.listdir(tmpdir)
                        if filename.endswith('.tif.aux.xml')]
        for filename in tmpfilenames:
            os.remove(os.path.join(tmpdir, filename))


class TestOtbFunctions(unittest.TestCase):

    def setUp(self):
        self.raster = Raster('tests/data/l8_20130714.tif')

    def test_should_remove_band(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        result = self.raster.remove_band(6, out_file.name)
        self.assertEqual(result.meta['count'], self.raster.meta['count'] - 1)
        _check_output_image(tester=self,
                            filename=out_file.name,
                            driver=u'GTiff',
                            width=self.raster.meta['width'],
                            height=self.raster.meta['height'],
                            number_bands=self.raster.meta['count'] - 1,
                            dtype=self.raster.meta['dtype'].ustr_dtype,
                            proj=self.raster.meta['srs'].ExportToProj4())

    def test_should_compute_ndvi(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.raster.ndvi(out_file.name, idx_red=4, idx_nir=5)
        _check_output_image(tester=self,
                            filename=out_file.name,
                            driver=u'GTiff',
                            width=self.raster.meta['width'],
                            height=self.raster.meta['height'],
                            number_bands=1,
                            dtype='Float32',
                            proj=self.raster.meta['srs'].ExportToProj4())

    def test_should_compute_ndwi(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.raster.ndwi(out_file.name, idx_nir=4, idx_mir=5)
        _check_output_image(tester=self,
                            filename=out_file.name,
                            driver=u'GTiff',
                            width=self.raster.meta['width'],
                            height=self.raster.meta['height'],
                            number_bands=1,
                            dtype='Float32',
                            proj=self.raster.meta['srs'].ExportToProj4())

    def test_should_compute_mndwi(self):
        out_file = tempfile.NamedTemporaryFile(suffix='.tif')
        self.raster.mndwi(out_file.name, idx_green=4, idx_mir=5)
        _check_output_image(tester=self,
                            filename=out_file.name,
                            driver=u'GTiff',
                            width=self.raster.meta['width'],
                            height=self.raster.meta['height'],
                            number_bands=1,
                            dtype='Float32',
                            proj=self.raster.meta['srs'].ExportToProj4())

    def test_lsms_segmentation_should_compute_segmented_image(self):
        # Compute and check first step (smoothing)
        out_file_filtered = tempfile.NamedTemporaryFile(suffix='.tif')
        out_file_spatial = tempfile.NamedTemporaryFile(suffix='.tif')
        filtered, spatial = self.raster.lsms_smoothing(
            output_filtered_image=out_file_filtered.name,
            spatialr=5,
            ranger=15,
            maxiter=5,
            thres=0.1,
            rangeramp=0,
            output_spatial_image=out_file_spatial.name)
        _check_output_image(tester=self,
                            filename=out_file_filtered.name,
                            driver=u'GTiff',
                            width=self.raster.meta['width'],
                            height=self.raster.meta['height'],
                            number_bands=self.raster.meta['count'],
                            dtype='Float32',
                            proj=self.raster.meta['srs'].ExportToProj4())
        _check_output_image(tester=self,
                            filename=out_file_spatial.name,
                            driver=u'GTiff',
                            width=self.raster.meta['width'],
                            height=self.raster.meta['height'],
                            number_bands=2,
                            dtype='Float32',
                            proj=self.raster.meta['srs'].ExportToProj4())

        # Compute and check second step (segmentation)
        out_file_segmented = tempfile.NamedTemporaryFile(suffix='.tif')
        segmented = filtered.lsms_seg(input_pos_img=spatial,
                                      output_seg_image=out_file_segmented.name,
                                      spatialr=5,
                                      ranger=15)
        _check_output_image(tester=self,
                            filename=out_file_segmented.name,
                            driver=u'GTiff',
                            width=self.raster.meta['width'],
                            height=self.raster.meta['height'],
                            number_bands=1,
                            dtype='Float32',
                            proj=self.raster.meta['srs'].ExportToProj4())

        # Compute and check third step (merging)
        out_file_segmented_merged = tempfile.NamedTemporaryFile(suffix='.tif')
        segmented_merged = segmented.lsms_merging(
            in_smooth=filtered,
            output_merged=out_file_segmented_merged.name,
            minsize=10)
        _check_output_image(tester=self,
                            filename=out_file_segmented_merged.name,
                            driver=u'GTiff',
                            width=self.raster.meta['width'],
                            height=self.raster.meta['height'],
                            number_bands=1,
                            dtype='Float32',
                            proj=self.raster.meta['srs'].ExportToProj4())

        # Compute and check fourth step (vectorization) by comparing:
        # number of labels and projections
        out_segmented_merged_shp_filename = os.path.join(tempfile.gettempdir(),
                                                         'segmented_merged.shp')
        segmented_merged.lsms_vectorisation(
            in_image=self.raster,
            output_vector=out_segmented_merged_shp_filename)

        ds = gdal.Open(segmented_merged.filename, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        array = np.array(band.ReadAsArray())
        number_labels = len(np.unique(array))

        ds = ogr.Open(out_segmented_merged_shp_filename)
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


class TestRealRaster(unittest.TestCase):

    def setUp(self):
        self.filename = 'tests/data/RGB.byte.tif'
        self.raster = Raster(self.filename)

    def test_should_get_attr_values_of_raster(self):
        self.assertEqual(self.raster.filename, self.filename)
        self.assertEqual(self.raster.meta['count'], 3)
        self.assertEqual(
            self.raster.meta['srs'].ExportToWkt(),
            'PROJCS["UTM Zone 18, Northern Hemisphere",GEOGCS["Unknown datum '
            'based upon the WGS 84 ellipsoid",DATUM["Not_specified_based_on'
            '_WGS_84_spheroid",SPHEROID["WGS 84",6378137,298.257223563,'
            'AUTHORITY["EPSG","7030"]]],PRIMEM["Greenwich",0],UNIT["degree",'
            '0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",'
            '-75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",'
            '500000],PARAMETER["false_northing",0],UNIT["metre",1,'
            'AUTHORITY["EPSG","9001"]]]')
        self.assertEqual(self.raster.meta['dtype'].lstr_dtype, 'uint8')
        self.assertEqual(self.raster.meta['driver'].GetDescription(), u'GTiff')
        self.assertEqual(self.raster.meta['transform'], (101985.0,
                                                         300.0379266750948,
                                                         0.0,
                                                         2826915.0,
                                                         0.0,
                                                         -300.041782729805))
        self.assertEqual(self.raster.meta['height'], 718)
        self.assertEqual(self.raster.meta['width'], 791)

    def test_should_get_array_of_raster(self):
        array = self.raster.array()
        self.assertEqual(array.ndim, 3)
        self.assertEqual(array.shape, (718, 791, 3))
        self.assertEqual(array.dtype, 'UInt8')

    def test_raster_should_raise_runtime_error_on_missing_file(self):
        filename = 'tests/data/not_exists.tif'
        self.assertRaises(RuntimeError, Raster, filename)


def suite():
    suite = unittest.TestSuite()
    load_from = unittest.defaultTestLoader.loadTestsFromTestCase
    suite.addTests(load_from(TestRealRaster))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
