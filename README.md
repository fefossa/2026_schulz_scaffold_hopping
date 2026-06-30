# 2026_schulz_scaffold_hopping

This repository contains the custom Cell Painting analysis scripts used in the manuscript:

**Structure-based Scaffold Hopping Reveals Strategies to Overcome Oncogenic KIT and PDGFRA Mutation-Driven Drug-Resistance in GIST**

T. Schulz, M. Beerbaum, A. Scrima, H. Jantzen, A. Teuber, T. Mühlenberg, L. Ebel, **F. Garcia-Fossa**, A. George, N. Berner, J. Weisner, M. P. Müller, S. Wilhelm, S. Sievers, S. Bauer, and D. Rauh.

Preprint: https://doi.org/10.21203/rs.3.rs-8307571/v1

## Contents

- `01_cellpainting_pca_analysis.ipynb` – main analysis notebook.
- `utils_cellpainting.py` – helper functions for filtering, aggregation, PCA and heatmap generation.
- `2026_03_config.yaml` – analysis configuration.
- `environment.yml` – Conda environment used for the analysis.

## Data

The notebook expects aggregated Cell Painting profiles as input. The raw imaging data are not included in this repository.

## Software

The analysis was performed using Python and the packages listed in `environment.yml`.

## License

MIT License.
