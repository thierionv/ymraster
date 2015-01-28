# -*- coding: utf-8 -*-

import functools


def fix_missing_proj(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kw):
        raster = method(self, *args, **kw)
        if raster.meta['srs'] is None \
                or not self.meta['srs'].IsSame(raster.meta['srs']):
            raster.set_projection(self.meta['srs'])
        return raster
    return wrapper
