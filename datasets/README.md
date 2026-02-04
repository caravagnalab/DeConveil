# Datasets

This directory contains example datasets derived from The Cancer Genome Atlas (TCGA), provided to facilitate testing of the DeConveil framework.
All datasets were downloaded from the Genomic Data Commons (GDC) portal using the `TCGAbiolinks` R package.

## TCGA-BRCA (breas cancer)

The `tcga_brca` folder contains matched tumor–normal breast cancer samples from the TCGA-BRCA cohort.
These data can be used to test the main copy-number–aware differential expression (DE) framework implemented in `DeConveil`.

The dataset can be loaded using the `load_test_data` function from `deconveil.utils_processing`.

Files:
- `rna.csv`: gene-level RNA-seq expression counts;
- `cnv.csv`: gene-level copy number variation (CNV) matrix aligned with RNA-seq matrix;
- `cnv_tumor.csv`: tumor-specific CNV matrix;
- `metadata.csv`: sample metadata, including tumor/normal status and sample identifier.

## TCGA-LUAD (lung adenocarcinoma)

The tcga_luad folder provides example data from the TCGA-LUAD cohort.
These data are intended to test the complementary Negative Binomial (NB) regression framework, implemented using the Stan-based model.

Files:
- `stan_joint_long.csv`: long-format input table for the NB regression Stan model;
- `results_nb.csv`: output results from the NB regression model.

## Folder organisation

```
DeConveil
│
└── datasets
    │
    ├── tcga_brca
    │   ├── rna.csv
    │   ├── cnv.csv
    │   ├── cnv_tumor.csv
    │   └── metadata.csv
    │
    └── tcga_luad
        ├── stan_joint_long.csv
        └── results_nb.csv
```
