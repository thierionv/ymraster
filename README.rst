Yet one More Raster library
===========================

The ``ymraster`` package contains tools for:

* manipulating raster images,
* perform classifications on these images.

For example, you can simply create a ``Raster`` instance and compute the NDVI
with the following simple commands.

>>> from ymraster import Raster
>>> raster = Raster('data/l8_20130425.tif')
>>> ndvi = raster.ndvi('l8_20130425_ndvi.tif', idx_red=4, idx_nir=5)


How to install
--------------

Prerequisites
`````````````

You need to have the following tools installed and properly set up:

* `Orfeo Toolbox <http://www.orfeo-toolbox.org/CookBook/>`_ (OTB) for most
  raster computations,
* `GDAL <http://gdal.org/>`_ for reading and writing rasters,
* `NumPy <http://www.numpy.org/>`_ for matrix & numeric computations,
* `scikit-learn <http://scikit-learn.org/>`_ for classifications.

Surely, there are already binary packages for these tools for your Linux
distribution.

Note that you have to set some environment variables in order for OTB to work.
Add the following lines to your ``~/.bashrc`` and adapt them to your
environment::

        export PYTHONPATH=${PYTHONPATH}${PYTHONPATH:+:}/usr/lib/otb/python
        export ITK_AUTOLOAD_PATH=/usr/lib/otb/applications


Installation
````````````

Simply clone the repository in a folder of your choice::

        $ cd </path/to/folder>  # eg. ~/.local/opt
        $ git clone https://github.com/ygversil/ymraster.git

Then install this into your environment.::

        $ cd ymraster
        $ pip install -e [--user] ./

Now you can import ``ymraster`` in Python.

>>> import ymraster
