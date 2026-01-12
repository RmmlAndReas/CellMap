import setuptools
import sys
from pathlib import Path

# Note: Archive directory and epyseg dependencies have been completely removed
# from version import __VERSION__  # Uncomment if version is needed

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='cellmap',
    version='1.0.0',
    description='CellMap - Cell segmentation and tracking software using Cellpose',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={'': ['*.md','*.json']},
    include_package_data=True,
    packages=setuptools.find_packages(),
    py_modules=['cellmap_main'],
    entry_points={
        'console_scripts': [
            'cellmap=cellmap_main:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        "cellpose>=2.0",
        "torch",
        "czifile",
        "Markdown",
        "matplotlib>=3.5.2",  # Required for PyQt6 support
        "numpy",
        "Pillow>=8.1.2",  # Security fix
        "PyQt6",  # May require: sudo apt install qt6-base-dev
        "read-lif",
        "scikit-image>=0.19.3",  # Required for 3D affine transformation
        "scipy>=1.7.3",  # Required for 3D affine transformation
        "scikit-learn>=1.0.2",  # Required for 3D affine transformation
        "tifffile>=2021.11.2",
        "tqdm",
        "natsort",
        "numexpr",
        "urllib3",  # For model download
        "qtawesome",  # For TA icons
        "pandas",
        "numba",
        "elasticdeform",  # Library for data augmentation
        "roifile",  # For ImageJ ROI support
        "prettytable",  # For SQL preview in pyTA
        "pyperclip",  # For pyTA lists
        "QtPy>=2.1.0",  # PyQt/PySide integration
        "deprecated",
        "tensorflow",  # Required for deep learning features
        "segmentation-models",  # Required for deep learning segmentation
    ],
    python_requires='>=3.6, <3.13',
)
