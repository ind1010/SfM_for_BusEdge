## Leveraging Structure from Motion to Localize Inaccessible Bus Stops
This repository contains code for the paper **Leveraging Structure from Motion to Localize Inaccessible Bus Stops**

*Authors: Indu Panigrahi, Tom Bu, and Christoph Mertz*
<br>

### Contents:
* ``BusEdge/`` - This directory contains scripts for obtaining and filtering images from the [BusEdge Platform](https://github.com/CanboYe/BusEdge).
* ``Mask2Former/`` - This directory contains the segmentation model that we use (Mask2Former) and scripts for detecting snow coverage. See the README for more information.
* ``colmap/`` - This directory contains sample images (references + query) and the associated reconstruction in the form of text files. These are used by the scripts in ``Mask2Former/``.
<br>

### How to use this repository:
1. Run COLMAP on clear weather images from a scene and then re-run COLMAP with the query added. Then, save the resulting reconstruction as text files using the naming conventions that we use in the sample files and folders that we provide in ``colmap/``.
2. Run ``Mask2Former/estimate_ground_plane.ipynb``. We provide sample output from this notebook in ``Mask2Former/sample_output/``.
3. Separate the query image from the clear weather images into the directory structure demonstrated in ``Mask2Former/images/``.
4. Run ``Mask2Former/detect_coverage.ipynb``. This notebook performs the detection of snow-covered sidewalks.
<br>

### Requirements:
* Used CUDA 11.1
* See ``requirements.txt``
<br>

### Reference:
If you find this code useful, please cite it as follows:
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
