# DeConveil 

<img src="docs/deconveil_logo.png" align="right" width="300">

#
[![pypi version](https://img.shields.io/pypi/v/DeConveil)](https://pypi.org/project/DeConveil)

## Introduction

The goal of *DeConveil* is the extension of Differential Gene Expression testing by accounting for genome aneuploidy.
This computational framework extends traditional DGE analysis by integrating DNA Copy Number Variation (CNV) data.
This approach adjusts for dosage effects and categorizes genes as *dosage-sensitive (DSG)*, *dosage-insensitive (DIG)*, and *dosage-compensated (DCG)*, separating the expression changes caused by CNVs from other alterations in transcriptional regulation.
To perform this gene separation we need to carry out DGE testing using both *PyDESeq2 (CN-naive)* and *DeConveil (CN-aware)* methods.

You can download the results of our analysis from [deconveilCaseStudies](https://github.com/kdavydzenka/deconveilCaseStudies)


## Installation

**Pre-required installations before running DeConveil**

Python libraries are required to be installed: *pydeseq2*

`pip install pydeseq2`

`DeConveil` can be installed from PyPI using `pip`:

`pip install DeConveil`

`DeConveil` can also be installed from Bioconda with `conda`:

`conda install -c bioconda deconveil`


## Data

**Input data**

DeConveil requires the following input matrices: 

    - matched mRNA read counts (normal and tumor samples) and absolute CN values (for normal diploid samples we assign CN=2), structured as NxG matrix, where N represents the number of samples and G represents the number of genes;
    
    - a design matrix structured as an N × F matrix, where N is the number of samples and F is the number of features or covariates.
    
Example of CN data for a given gene *g*:
CN = [1, 2, 3, 4, 5, 6].

An example of the input data can be found in the *test_deconveil* Jupyter Notebook.


**Output data**

`res_CNnaive.csv` (for *PyDESeq2* method) and `res_CNaware.csv` (for *DeConveil*) data frames reporting *log2FC* and *p.adjust* values for both methods.

These data frames are further processed to separate gene groups using `define_gene_groups()` function included in DeConveil framework.

A tutorial of the analysis workflow is available in `test_deconveil.ipynb`


#### Citation

[![](http://img.shields.io/badge/doi-10.1101/2025.03.29.646108-red.svg)](https://doi.org/10.1101/2025.03.29.646108)

If you use `DeConveil`, cite:

K. Davydzenka, G. Caravagna, G. Sanguinetti. Extending differential gene expression testing to handle genome aneuploidy in cancer. [bioRxiv preprint](https://doi.org/10.1101/2025.03.29.646108), 2025.


#### Copyright and contacts

Katsiaryna Davydzenka, Cancer Data Science (CDS) Laboratory.

[![](https://img.shields.io/badge/CDS%20Lab%20Github-caravagnalab-seagreen.svg)](https://github.com/caravagnalab)
[![](https://img.shields.io/badge/CDS%20Lab%20webpage-https://www.caravagnalab.org/-red.svg)](https://www.caravagnalab.org/)



