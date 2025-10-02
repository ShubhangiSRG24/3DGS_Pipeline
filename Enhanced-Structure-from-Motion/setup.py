#!/usr/bin/env python3
"""
Setup script for Enhanced SfM Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="enhanced-sfm",
    version="0.1.0",
    description="GPU-accelerated Structure-from-Motion pipeline with modern feature extractors and matchers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Minhyeok Im",
    author_email="minhyeok0104@gmail.com",
    url="https://github.com/yourusername/Enhanced-Structure-from-Motion",
    packages=find_packages(),
    py_modules=["sfm_pipeline"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "kornia>=0.6.12",
        "lightglue @ git+https://github.com/cvg/LightGlue.git",
        "numba>=0.58.0",
        "psutil>=5.9.0",
        "h5py>=3.8.0",
        "transformers>=4.30.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "gpu": [
            "cupy-cuda12x>=12.0.0",  # For CUDA 12.x
            "faiss-gpu>=1.7.0",
            "pyceres>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sfm-pipeline=sfm_pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    keywords="structure-from-motion, computer-vision, 3d-reconstruction, feature-matching",
) 