Yet one More Raster library
===========================

The ``ymraster`` package contains tools for manipulating raster images.

Simply create a ``Raster`` instance giving the filename of the raster you want to read.

>>> raster = Raster('data/l8_20130425.tif')

You can then for example compute the NDVI.

>>> ndvi = raster.ndvi(idx_red=4, idx_nir=5)
>>> ndvi.meta['count'] = 1


How to install
--------------

Prerequisites
`````````````

You need to have the following tools installed and properly set up:

* `Orfeo Toolbox <http://www.orfeo-toolbox.org/CookBook/>`_ (OTB) for most
  raster computations,
* `GDAL <http://gdal.org/>`_ for reading and writing rasters metadata,
* `NumPy <http://www.numpy.org/>`_ for matrix & numeric computations,
* `rasterio <https://github.com/mapbox/rasterio>`_ for reading and saving
  rasters efficiently.

The first three are most certainly packaged in your Linux distribution. For the last one use pip.::

        pip install --user rasterio

Note that you have to set some environment variables in order for OTB to work.
Add the following lines to your ``.bashrc`` or adapt them to your environment::

        export PYTHONPATH=${PYTHONPATH}${PYTHONPATH:+:}/usr/lib/otb/pyhon
        export ITK_AUTOLOAD_PATH=/usr/lib/otb/applications


Installation
````````````

Simple clone the repository in a folder of your choice::

        cd </path/to/folder>  # eg. ~/.local/opt
        git clone https://github.com/ygversil/ymraster.git

Then create a .pth file in your ``site-packages`` folder with the path to the folder.::

        cd /.local/lib/python2.7/site-pakcages  # create directory if it does not exists
        echo "/home/<user>/local/opt/ymraster" > ymraster.pyh

Now you can import ``ymraster`` in Python.

>>> import ymraster
>>>
