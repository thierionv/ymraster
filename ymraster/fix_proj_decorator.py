# -*- coding: utf-8 -*-

import functools


def fix_missing_proj(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kw):
        raster = method(self, *args, **kw)
        if raster.srs is None \
                or not self.srs.IsSame(raster.srs):
            raster.srs = self.srs
        return raster
    return wrapper
