"""Generic dialog for displaying analysis results."""

import os
import csv
from typing import Dict, List, Optional
import numpy as np

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel,
    QPushButton, QDialogButtonBox, QTableWidget, QTableWidgetItem,
    QScrollArea, QTextEdit, QFileDialog, QMessageBox
)
from qtpy.QtCore import Qt
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    # Fallback for older matplotlib versions
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    except ImportError:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui.analysis.analysis_results import AnalysisResults
from utils.logger import TA_logger

logger = TA_logger()


class AnalysisResultsDialog(QDialog):
    """Dialog for displaying analysis results."""
    
    def __init__(self, results: AnalysisResults, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle(f"Analysis Results: {results.analysis_name}")
        self.setMinimumSize(800, 600)
        self.result_directory = None  # Store the result directory path
        self.init_ui()
        # Automatically save results when dialog is created
        self._auto_save_results()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        
        # Create tab widget for different result types
        tabs = QTabWidget()
        
        # Figures tab
        if self.results.figures:
            figures_widget = self._create_figures_widget()
            tabs.addTab(figures_widget, "Plots")
        
        # Images tab
        if self.results.images:
            images_widget = self._create_images_widget()
            tabs.addTab(images_widget, "Images")
        
        # Data tab
        if self.results.data:
            data_widget = self._create_data_widget()
            tabs.addTab(data_widget, "Data")
        
        # Metadata tab
        if self.results.metadata:
            metadata_widget = self._create_metadata_widget()
            tabs.addTab(metadata_widget, "Info")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_box = QHBoxLayout()
        
        save_button = QPushButton("Save Results...")
        save_button.clicked.connect(self.save_results)
        button_box.addWidget(save_button)
        
        # Show result directory button (only if results were auto-saved)
        if self.result_directory and os.path.exists(self.result_directory):
            show_dir_button = QPushButton("Show Result Directory")
            show_dir_button.clicked.connect(self.show_result_directory)
            button_box.addWidget(show_dir_button)
        
        button_box.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_box.addWidget(close_button)
        
        layout.addLayout(button_box)
        
        self.setLayout(layout)
    
    def _create_figures_widget(self) -> QWidget:
        """Create widget for displaying matplotlib figures."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        if len(self.results.figures) == 1:
            # Single figure - display directly
            canvas = FigureCanvasQTAgg(self.results.figures[0])
            layout.addWidget(canvas)
        elif len(self.results.figures) == 2:
            # Two figures - display side by side
            from qtpy.QtWidgets import QHBoxLayout
            h_layout = QHBoxLayout()
            for fig in self.results.figures:
                canvas = FigureCanvasQTAgg(fig)
                h_layout.addWidget(canvas)
            layout.addLayout(h_layout)
        else:
            # Multiple figures - use tabs or scroll
            figure_tabs = QTabWidget()
            for i, fig in enumerate(self.results.figures):
                canvas = FigureCanvasQTAgg(fig)
                figure_tabs.addTab(canvas, f"Plot {i+1}")
            layout.addWidget(figure_tabs)
        
        widget.setLayout(layout)
        return widget
    
    def _create_images_widget(self) -> QWidget:
        """Create widget for displaying images."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        images_container = QWidget()
        images_layout = QVBoxLayout()
        
        for name, image_array in self.results.images.items():
            label = QLabel(f"{name}:")
            images_layout.addWidget(label)
            
            # Convert numpy array to QPixmap
            from qtpy.QtGui import QPixmap, QImage
            from utils.image_utils import toQimage
            
            qimage = toQimage(image_array)
            pixmap = QPixmap.fromImage(qimage)
            
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            images_layout.addWidget(image_label)
        
        images_container.setLayout(images_layout)
        scroll.setWidget(images_container)
        layout.addWidget(scroll)
        
        widget.setLayout(layout)
        return widget
    
    def _create_data_widget(self) -> QWidget:
        """Create widget for displaying data tables."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Create tabs for different data types
        data_tabs = QTabWidget()
        
        for data_name, data in self.results.data.items():
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # List of dictionaries - create table
                table = QTableWidget()
                table.setRowCount(len(data))
                
                # Get column names from first row
                columns = list(data[0].keys())
                table.setColumnCount(len(columns))
                table.setHorizontalHeaderLabels(columns)
                
                # Populate table
                for row_idx, row_data in enumerate(data):
                    for col_idx, col_name in enumerate(columns):
                        value = row_data.get(col_name, "")
                        item = QTableWidgetItem(str(value))
                        table.setItem(row_idx, col_idx, item)
                
                table.resizeColumnsToContents()
                data_tabs.addTab(table, data_name)
            else:
                # Other data types - show as text
                text_widget = QTextEdit()
                text_widget.setReadOnly(True)
                text_widget.setText(str(data))
                data_tabs.addTab(text_widget, data_name)
        
        layout.addWidget(data_tabs)
        widget.setLayout(layout)
        return widget
    
    def _create_metadata_widget(self) -> QWidget:
        """Create widget for displaying metadata."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        
        metadata_text = "Analysis Metadata:\n\n"
        for key, value in self.results.metadata.items():
            metadata_text += f"{key}: {value}\n"
        
        text_widget.setText(metadata_text)
        layout.addWidget(text_widget)
        
        widget.setLayout(layout)
        return widget
    
    def _auto_save_results(self):
        """Automatically save results to Analysis/ folder."""
        # Get TA output folder from metadata
        ta_output_folder = self.results.metadata.get('ta_output_folder')
        if not ta_output_folder:
            logger.warning("ta_output_folder not in metadata, skipping auto-save")
            return
        
        # Use Analysis/ subfolder in TA output folder
        analysis_folder = os.path.join(ta_output_folder, "Analysis")
        
        # Create Analysis folder if it doesn't exist
        try:
            os.makedirs(analysis_folder, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating Analysis folder: {e}")
            return
        
        try:
            # Create a timestamped subfolder for this analysis run
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_name_safe = self.results.analysis_name.replace(" ", "_").lower()
            run_folder = os.path.join(analysis_folder, f"{analysis_name_safe}_{timestamp}")
            os.makedirs(run_folder, exist_ok=True)
            
            # Save figures
            for i, fig in enumerate(self.results.figures):
                fig_path = os.path.join(run_folder, f"plot_{i+1}.png")
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved plot to {fig_path}")
            
            # Save images
            for name, image_array in self.results.images.items():
                from utils.image_io import Img
                # Sanitize name for filename
                safe_name = name.replace(" ", "_").replace("/", "_").lower()
                img_path = os.path.join(run_folder, f"{safe_name}.tif")
                Img(image_array, dimensions='hwc').save(img_path)
                logger.info(f"Saved image {name} to {img_path}")
            
            # Save CSV data
            for data_name, data in self.results.data.items():
                if isinstance(data, str) and data_name.lower().endswith('.csv'):
                    # Already CSV string
                    safe_name = data_name.replace(" ", "_").replace("/", "_").lower()
                    if not safe_name.endswith('.csv'):
                        safe_name += '.csv'
                    csv_path = os.path.join(run_folder, safe_name)
                    with open(csv_path, 'w', newline='') as f:
                        f.write(data)
                    logger.info(f"Saved CSV data to {csv_path}")
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # List of dictionaries - convert to CSV
                    safe_name = data_name.replace(" ", "_").replace("/", "_").lower()
                    if not safe_name.endswith('.csv'):
                        safe_name += '.csv'
                    csv_path = os.path.join(run_folder, safe_name)
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    logger.info(f"Saved CSV data to {csv_path}")
            
            logger.info(f"Analysis results automatically saved to: {run_folder}")
            # Store the result directory path
            self.result_directory = run_folder
        except Exception as e:
            logger.error(f"Error auto-saving results: {e}")
            import traceback
            traceback.print_exc()
    
    def save_results(self):
        """Save analysis results to disk in Analysis/ folder."""
        # Get TA output folder from metadata
        ta_output_folder = self.results.metadata.get('ta_output_folder')
        if not ta_output_folder:
            # Fallback: ask user to select folder
            folder = QFileDialog.getExistingDirectory(self, "Select folder to save results")
            if not folder:
                return
            analysis_folder = os.path.join(folder, "Analysis")
        else:
            # Use Analysis/ subfolder in TA output folder
            analysis_folder = os.path.join(ta_output_folder, "Analysis")
        
        # Create Analysis folder if it doesn't exist
        try:
            os.makedirs(analysis_folder, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating Analysis folder: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create Analysis folder: {e}")
            return
        
        try:
            # Create a timestamped subfolder for this analysis run
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_name_safe = self.results.analysis_name.replace(" ", "_").lower()
            run_folder = os.path.join(analysis_folder, f"{analysis_name_safe}_{timestamp}")
            os.makedirs(run_folder, exist_ok=True)
            
            # Save figures
            for i, fig in enumerate(self.results.figures):
                fig_path = os.path.join(run_folder, f"plot_{i+1}.png")
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved plot to {fig_path}")
            
            # Save images
            for name, image_array in self.results.images.items():
                from utils.image_io import Img
                # Sanitize name for filename
                safe_name = name.replace(" ", "_").replace("/", "_").lower()
                img_path = os.path.join(run_folder, f"{safe_name}.tif")
                Img(image_array, dimensions='hwc').save(img_path)
                logger.info(f"Saved image {name} to {img_path}")
            
            # Save CSV data
            for data_name, data in self.results.data.items():
                if isinstance(data, str) and data_name.lower().endswith('.csv'):
                    # Already CSV string
                    safe_name = data_name.replace(" ", "_").replace("/", "_").lower()
                    if not safe_name.endswith('.csv'):
                        safe_name += '.csv'
                    csv_path = os.path.join(run_folder, safe_name)
                    with open(csv_path, 'w', newline='') as f:
                        f.write(data)
                    logger.info(f"Saved CSV data to {csv_path}")
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # List of dictionaries - convert to CSV
                    safe_name = data_name.replace(" ", "_").replace("/", "_").lower()
                    if not safe_name.endswith('.csv'):
                        safe_name += '.csv'
                    csv_path = os.path.join(run_folder, safe_name)
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    logger.info(f"Saved CSV data to {csv_path}")
            
            QMessageBox.information(self, "Success", f"Results saved to:\n{run_folder}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save results: {e}")
    
    def show_result_directory(self):
        """Open the result directory in the system file manager."""
        if not self.result_directory or not os.path.exists(self.result_directory):
            QMessageBox.warning(self, "Directory Not Found", 
                              "Result directory not found or not yet created.")
            return
        
        try:
            import platform
            import subprocess
            
            system = platform.system()
            if system == "Windows":
                os.startfile(self.result_directory)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", self.result_directory])
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", self.result_directory])
        except Exception as e:
            logger.error(f"Error opening result directory: {e}")
            QMessageBox.warning(self, "Error", 
                              f"Could not open result directory:\n{self.result_directory}\n\n"
                              f"Error: {e}\n\n"
                              f"You can manually navigate to:\n{self.result_directory}")