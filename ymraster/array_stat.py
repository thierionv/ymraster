# -*- coding: utf-8 -*-

import numpy as np

from collections import defaultdict

_STATS = [
    'min',
    'max',
    'mean',
    'std',
    'median',
    'quartile1',
    'quartile3',
    'percentile',
]

_SUMMARY_STAT_FUNC = {}
_SUMMARY_STAT_FUNC['min'] = np.argmin
_SUMMARY_STAT_FUNC['max'] = np.argmax

_STAT_FUNC = defaultdict(lambda: np.percentile)
_STAT_FUNC['min'] = np.nanmin
_STAT_FUNC['max'] = np.nanmax
_STAT_FUNC['mean'] = np.nanmean
_STAT_FUNC['std'] = np.nanstd
_STAT_FUNC['median'] = np.median

_COMMON_PERCENTILES = {
    'quartile1': 25,
    'quartile3': 75,
}


class ArrayStat(object):
    """Represent a stat that is computable on a NumPy array"""

    def __init__(self, s, axis=None):
        self.stat = s if ':' not in s else s.split(':')[0]
        if self.stat not in _STATS:
            raise ValueError("Not a recognized statistic: {}".format(s))
        self.percentage = None if ':' not in s else float(s.split(':')[1])
        self.func = _STAT_FUNC[s]
        self.is_summary = s in _SUMMARY_STAT_FUNC
        self.summary_func = _SUMMARY_STAT_FUNC[s] if self.is_summary else None
        if self.func is np.percentile and self.percentage is None:
            try:
                self.percentage = _COMMON_PERCENTILES[s]
            except KeyError:
                raise ValueError("Missing percentage for: {}".format(s))
        self.axis = axis

    def compute(self, array):
        kw = {'a': array}
        if self.axis is not None:
            kw['axis'] = self.axis
        if self.percentage is not None:
            kw['q'] = self.percentage

        return self.func(**kw)

    def indices(self, array):
        kw = {'a': array}
        if self.axis is not None:
            kw['axis'] = self.axis

        return self.summary_func(**kw)
