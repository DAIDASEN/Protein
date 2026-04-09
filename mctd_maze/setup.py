"""
Setup script for mctd_maze package.

Install with:
    pip install -e .
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mctd_maze",
    version="0.1.0",
    author="MCTD-Maze Implementation",
    description=(
        "Monte Carlo Tree Diffusion for System 2 Planning (OGBench)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "mujoco>=3.1.0",
        "ogbench>=1.0.0",
        "einops>=0.7.0",
        "tqdm>=4.65",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "black>=23.0",
            "isort>=5.12",
        ],
        "full": [
            "wandb>=0.17.0",
            "hydra-core>=1.3.2",
            "stable-baselines3>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mctdmaze-train=scripts.train:main",
            "mctdmaze-eval=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
