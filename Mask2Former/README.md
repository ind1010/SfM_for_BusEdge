Files:
* ``mask2former/`` and ``configs/`` are provided from the original [Mask2Former GitHub](https://github.com/facebookresearch/Mask2Former).
* ``images/`` is split into 2 directories: ``clear_references/`` contains clear-weather references images and ``query/`` contains a sample query.
* ``estimate_ground_plane.ipynb`` is the Jupyter notebook that is used to estimate the ground plane and produce corresponding text files.
* ``sample_output`` contains sample output from ``estimate_ground_plane.ipynb``.
* ``detect_coverage.ipynb`` is the Jupyter notebook that uses the output of ``estimate_ground_plane.ipynb`` to classify a query image as a Snow-covered sidewalk or a Clear sidewalk.
* The ``.py`` files are helper files for the notebooks.
