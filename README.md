# CellMap

CellMap is a Python-based software for cell segmentation and tracking in time-lapse microscopy images.

## Features

- **Cellpose segmentation** - Automatic cell segmentation using Cellpose
- **Cell tracking** - Track cells across time frames
- **Cell measurements** - Measure cell properties (area, shape, orientation, etc.)
- **3D analysis** - 3D surface projection and measurements
- **SQLite database** - Store and query measurement data
- **Graphical user interface** - Easy-to-use GUI built with PyQt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RmmlAndReas/CellMap
cd CellMap
```

2. Install [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) (if not already installed)

3. Create and activate conda environment from the environment file:
```bash
conda env create -f environment.yml
conda activate CellMap
```

## Usage

Run the GUI:
```bash
python -m cellmap
```

Or after installation:
```bash
cellmap
```

## Workflow

1. **Load Images** - Add images to the file list
2. **Segment** - Use Cellpose to segment cells
3. **Track** - Track cells across time frames
4. **Measure** - Extract cell properties
5. **Analyze** - Query database and export results

## Dependencies

- cellpose>=2.0
- torch
- PyQt5/PyQt6
- numpy, scipy, scikit-image
- pandas
- tifffile, czifile, read-lif
- And other scientific computing libraries
