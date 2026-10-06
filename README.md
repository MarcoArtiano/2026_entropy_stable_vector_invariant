# Entropy-Stable and Well-Balanced Discontinuous Galerkin Methods for the Compressible Euler Equations in Vector-Invariant Form

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23144287.svg)](https://zenodo.org/doi/10.5281/zenodo.23144287)

This repository contains information and code to reproduce the results presented in the article

```bibtex
@online{artiano2026vectorinvariant,
  title={Entropy-Stable and Well-Balanced Discontinuous Galerkin Methods
         for the Compressible Euler Equations in Vector-Invariant Form},
  author={Artiano, Marco and Ricardo, Kieran and Knoth, Oswald and Spichtinger, Peter
          and Ranocha, Hendrik},
  year={2026},
  month={10},
  eprint={2610.05820},
  eprinttype={arxiv},
  eprintclass={math.NA},
  doi={10.48550/arXiv.2610.05820}
}
```

If you find these results useful, please cite the article mentioned above. If you use the implementations provided here, please also cite this repository as

```bibtex
@misc{artiano2026vectorinvariantRepo,
  title={Reproducibility repository for
         "{E}ntropy-Stable and Well-Balanced Discontinuous Galerkin Methods
          for the Compressible Euler Equations in Vector-Invariant Form"},
  author={Artiano, Marco and Ricardo, Kieran and Knoth, Oswald and Spichtinger, Peter
            and Ranocha, Hendrik},
  year={2026},
  howpublished={\url{https://github.com/MarcoArtiano/2026_entropy_stable_vector_invariant}},
  doi={10.5281/zenodo.23144287}
}
```

## Abstract
We develop structure-preserving methods for the compressible Euler equations with gravity in vector-invariant form and potential temperature as a prognostic variable within the flux-differencing discontinuous Galerkin spectral element method (DGSEM) framework.
By discretizing the nonconservative terms as symmetric and antisymmetric products, we derive two-point numerical fluxes that conserve both the thermodynamic entropy and the total energy.
Moreover, we design an entropy-stable numerical flux that is well-balanced for both isothermal and isentropic background states.
All properties are shown to carry over to the high-order DGSEM on general curvilinear meshes.
Several numerical examples confirm the theoretical findings and show the robustness and accuracy of the scheme for use in modern dynamical cores for atmospheric flows.

## Numerical experiments
To reproduce the numerical experiments presented in this article, you need to install Julia. The numerical experiments presented in this article were performed using Julia v1.10.6.

First, you need to download this repository, e.g., by cloning it with git or by downloading an archive via the GitHub interface. Then, you need to start Julia in the code directory of this repository and follow the instructions described in the README.md file therein.

## Authors
- Marco Artiano
- Kieran Ricardo
- Oswald Knoth
- Peter Spichtinger
- [Hendrik Ranocha](https://ranocha.de/) (Johannes Gutenberg University Mainz, Germany)

## License
The code in this repository is published under the MIT license, see the LICENSE file.

## Disclaimer
Everything is provided as is and without warranty. Use at your own risk!
