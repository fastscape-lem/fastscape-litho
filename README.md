# fastscape_litho

[![DOI](https://zenodo.org/badge/342188636.svg)](https://zenodo.org/badge/latestdoi/342188636)

Feeding fastscape with a 3D bloc of lithology with differential erodibility, diffusion and other. The input requires one 3D matrix of integer labels (0,1,2,3,4,...) and a list of label correspondence for each rock type specific parameters. The current version of the package includes basic functions to run erosion through a 3D bloc from a flat surface (see features). Future versions will include more feature, it highly depends how fast projects related to this components arises. Feel free to contact me if you are interested to see a specific feature getting developed! 

* Free software: BSD license
<!-- * Documentation: https://fastscape-litho.readthedocs.io. -->

# Quick Start

You need to install [`fastscape`](https://github.com/fastscape-lem/fastscape), with all dependencies and all. Then, you can simply:

```
pip install fastscape-litho
```

Note that you need to be in the fastscape environment. `conda-forge` package to come!


# Features

- 3D input of parameters
- 3D K with the basic stream power law (implementation from [Braun and Willett, 2013](https://doi.org/10.1016/j.geomorph.2012.10.008))
- 3D Kd for rocks and soil for hillslope linear diffusion

# Features to come:

- Initialization with existing topography -> *soon*
- Allow different geometry for the 3D bloc (e.g. different dx and dy) -> *soon*
- 3D rock density for flexure and isostasy -> **Kind Of Here But Deserves More Thoughts**
- 3D infiltration compatible with 2D precipitation -> **WIP**
- landscape parametrization (template of geological 3D blocs) -> *future*
- Bridge between [`gempy`](https://www.gempy.org/) and `fastscape-litho` -> *future*


# Credits

Boris Gailleton - boris.gailleton@gfz-potsdam.de

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
