# -*- coding: utf-8 -*-

import argparse
from ymraster import Raster, temporal_stats

from datetime import datetime
from functools import partial


def _valid_date(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return datetime.strptime(s, '%Y-%m-%d')
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def _days_between(dt0, dt):
    """Returns number of days between the two given datetime objects."""
    delta = dt - dt0
    return delta.days


def command_line_args():
    parser = argparse.ArgumentParser(
        description="Compute pixel-wise statistics from a given list of "
        "temporally distinct, but spatially identical, images. Only one band "
        "in each image is considered (by default: the first one).\n\n"
        "Output is a multi-band image where each band contains a result (eg. "
        "maximum, mean). For results taken from original values (eg. maximum), "
        "there is an additional band which gives the date/time at which the "
        "result has been found in numeric format (eg. maximum has occured on "
        "Apr 25, 2013 (midnight) -> 1366840800.0)")
    parser.add_argument("raster", nargs='+',
                        help="Space separated list of rasters")
    parser.add_argument("-s", "--stat", nargs='+',
                        help="Space separated list of statistics to compute")
    parser.add_argument("-b", "--band_idx", type=int, default=1,
                        help="Index of the band to compute statistics on, "
                        "the same for all images (default: 1)")
    parser.add_argument("-d", "--date_from", type=_valid_date,
                        help="If specified, values of bands which contains "
                        "statistics dates are number of days between the "
                        "statistic date and this given date (instead of the "
                        "found date in numeric format.\n\n"
                        "Must be given in format "
                        "YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'")
#    parser.add_argument("-t", "--time_raster",
#                       help="Path to a mono-band raster with same extent than "
#                        "the input images and whose values are dates in "
#                        "numeric format. If specified, statistical dates are "
#                        "computed relative to the dates in this raster")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. "
                        "A default name is chosen if omitted.")
    return parser.parse_args()


def compute_temporal_stats(args):
    rasters = [Raster(filename) for filename in args.raster]
    kw = {}
    kw['out_filename'] = args.out_file
    if args.band_idx is not None:
        kw['band_idx'] = args.band_idx
    if args.stat is not None:
        kw['stats'] = list(args.stat)
    if args.date_from is not None:
        kw['date2float'] = partial(_days_between, args.date_from)
    temporal_stats(*rasters, **kw)


def main():
    args = command_line_args()
    compute_temporal_stats(args)


if __name__ == "__main__":
    main()
