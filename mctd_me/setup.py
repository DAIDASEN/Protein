"""
Setup script for mctd_me package.

Install with:
    pip install -e .
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mctd_me",
    version="0.1.0",
    author="MCTD-ME Implementation",
    description=(
        "Monte Carlo Tree Diffusion with Multiple Experts for Protein Design"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "biopython>=1.81",
        "transformers>=4.38.0",
        "huggingface-hub>=0.21.0",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.6",
        ],
        "full": [
            "fair-esm>=2.0.0",
            "biotite>=0.38.0",
            "wandb>=0.16.0",
            "datasets>=2.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mctdme-inv=scripts.run_inverse_folding:main",
            "mctdme-fold=scripts.run_folding:main",
            "mctdme-motif=scripts.run_motif_scaffolding:main",
            "mctdme-eval=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
