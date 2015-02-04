# -*- coding: utf-8 -*-

import numpy as np

_STATS = [
    'min',
    'max',
    'mean',
    'std',
    'median',
    'quartile1'
    'quartile3'
    'percentile',
]

_STAT_FUNC = {
    'min': np.nanmin,
    'max': np.nanmax,
    'mean': np.nanmean,
    'std': np.nanstd,
    'median': np.median,
}

_SUMMARY_STATS = {
    'min': 0,
    'quartile1': 25,
    'median': 50,
    'quartile3': 75,
    'max': 100,
}


class NpStatFunc(object):
    """Represent a stat that is computable with a simple NumPy function or a
    combination of NumPy function
    """

    def __init__(self, s, percentage=None):
        if s not in _STATS:
            raise ValueError("Not a recognized statistic: {}".format(s))
        if s == 'percentile' and percentage is None:
            raise ValueError("Missing percentage for statistic: {}".format(s))
        self.stat = s
        try:
            self.func = _STAT_FUNC[s]
        except KeyError:
            self.func = np.percentile
        if self.func is np.percentile:
            try:
                self.percentage = _SUMMARY_STATS[s]
            except KeyError:
                self.percentage = percentage
        self.is_summary = s in _SUMMARY_STATS.keys() or s == 'percentile'
