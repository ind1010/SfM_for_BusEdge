## Leveraging Structure from Motion to Localize Inaccessible Bus Stops

Authors: Indu Panigrahi, Tom Bu, Christoph Mertz

Contents:
* ``BusEdge/`` - This directory contains scripts for obtaining and filtering images from the [BusEdge Platform](https://github.com/CanboYe/BusEdge).
* ``Mask2Former/`` - This directory contains the segmentation model that we use (Mask2Former) and scripts for detecting snow coverage. See the README for more information.
* ``colmap/`` - This directory contains sample images (references + query) and the associated reconstruction in the form of text files. These are used by the scripts in ``Mask2Former/``.


Requirements:
* Used CUDA 11.1
* See ``requirements.txt``
