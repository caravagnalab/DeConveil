from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="DeConveil",
    version="0.2.0",
    description="An extension of PyDESeq2/DESeq2 designed to account for genome aneuploidy",
    url="https://github.com/caravagnalab/DeConveil",
    author="Katsiaryna Davydzenka",
    author_email="katiasen89@gmail.com",
    license="MIT",
    packages=["deconveil"],
    python_requires=">=3.10.0",  
    install_requires=[
        "anndata>=0.8.0",
        "formulaic>=1.0.2",
        "numpy>=1.23.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.11.0",
        "formulaic-contrasts>=0.2.0",
        "matplotlib>=3.6.2",
        "seaborn>=0.12.2",
        "pydeseq2>=0.4.12",  
        "rpy2>=3.5.0",
    ],
    extras_require={
        # Optional Stan-based NB regression
        "stan": ["cmdstanpy>=1.2.0"],
        "dev": [
            "pytest>=6.2.4",
            "pre-commit>=2.13.0",
            "numpydoc",
            "coverage",
            "mypy",
            "pandas-stubs",
        ]  
    },
)

