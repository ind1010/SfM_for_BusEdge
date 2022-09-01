## Leveraging Structure from Motion to Localize Inaccessible Bus Stops
Authors: Indu Panigrahi, Tom Bu, and Christoph Mertz
<br>

### Contents:
* ``BusEdge/`` - This directory contains scripts for obtaining and filtering images from the [BusEdge Platform](https://github.com/CanboYe/BusEdge).
* ``Mask2Former/`` - This directory contains the segmentation model that we use ([Mask2Former](https://github.com/facebookresearch/Mask2Former)) and scripts for detecting snow coverage. See [``Mask2Former/README.md``](https://github.com/ind1010/SfM_for_BusEdge/blob/c7cb933717475d78cfac8ad0d290e826db85e23e/Mask2Former/README.md) for more information.
* ``colmap/`` - This directory contains sample images (references + query) and the associated reconstruction in the form of text files. These are used by the scripts in ``Mask2Former/``.

### Instructions:
[Here is a dataset of the images used in this work and along with images in additional categories](https://www.kaggle.com/datasets/indupanigrahi/busedge-sidewalks-and-more)
1. Run [COLMAP](https://colmap.github.io/) on clear weather images from a scene and then re-run COLMAP with the query added. Save the resulting reconstruction as text files using the naming conventions that we use in the sample files and folders that we provide in ``colmap/``.
2. Run ``Mask2Former/estimate_ground_plane.ipynb``. We provide sample output from this notebook in ``Mask2Former/sample_output/``.
3. Separate the query image from the clear weather images into the directory structure demonstrated in ``Mask2Former/images/``.
4. Run ``Mask2Former/detect_coverage.ipynb``. This notebook performs the detection of snow-covered sidewalks.

### Requirements:
* Used CUDA 11.1
* See ``requirements.txt``

### Reference:
Please cite this code as follows:
```
@online{panigrahi2022riss,
  author = {Indu Panigrahi and Tom Bu and Christoph Mertz},
  title = {{Leveraging Structure from Motion to Localize Inaccessible Bus Stops}},
  year = 2022,
  url = https://github.com/ind1010/SfM_for_BusEdge,
}


@mastersthesis{bu2022crosswalk,
author = {Tom Bu},
title = {{Towards HD Map Updates with Crosswalk Change Detection from Vehicle-Mounted Cameras}},
year = {2022},
month = {August},
school = {Carnegie Mellon University},
address = {Pittsburgh, PA},
number = {CMU-RI-TR-22-34},
}
```
