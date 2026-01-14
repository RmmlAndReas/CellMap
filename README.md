# CellMap

CellMap is a Python-based software for cell segmentation and tracking in time-lapse microscopy images.

## Features

- **Cellpose segmentation** - Automatic cell segmentation using Cellpose
- **Cell tracking** - Track cells across time frames
- **Cell measurements** - Measure cell properties (area, shape, orientation, etc.)
- **3D analysis** - 3D surface projection and measurements
- **SQLite database** - Store and query measurement data
- **Graphical user interface** - Easy-to-use GUI built with PyQt

## Installation & Integration

To integrate CellMap into your workflow, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RmmlAndReas/CellMap
   cd CellMap
   ```

2. **Install [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)** (if not already installed).

   *Note for Windows users:* Microsoft Visual C++ 14.0 or greater is required. Download it here: https://visualstudio.microsoft.com/visual-cpp-build-tools/

3. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate CellMap
   ```

4. **Integrate CellMap into your analysis:**  
   After activation, you can launch CellMap via the command line or import it as a Python package within your own scripts:
   ```bash
   python -m cellmap
   ```
   or inside Python:
   ```python
   import cellmap
   # Initialize and use CellMap functionality in your pipeline
   ```

CellMap can now be integrated into your data analysis pipelines or used as a standalone application.

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
