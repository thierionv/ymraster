# -*- coding: utf-8 -*-

import argparse
from ymraster import Raster


def compute_ndvi():
    # Command-line parameters
    parser = argparse.ArgumentParser(
        description="Compute Normalized Difference Water Index (NDWI) "
        "of the input image and save it into the specified output file. "
        "Indices of near-infrared and middle-infrared bands must be given"
        "(indices start at 1).")
    parser.add_argument("-i", "--in_file", help="Path to the input image",
                        required=True)
    parser.add_argument("-nir", "--idx_nir", help="Index of the near-infrared "
                        "band", type=int, required=True)
    parser.add_argument("-mir", "--idx_mir", help="Index of the middle-infrared"
                        " band", type=int, required=True)
    parser.add_argument("-o", "--out_file", help="Path to the output file",
                        type=int, required=True)
    args = parser.parse_args()

    # Do the actual work
    raster = Raster(args.in_file)
    raster.ndvi(args.out_file, args.idx_red, args.idx_nir)


if __name__ == "__main__":
    compute_ndvi()
