"""Analysis tab for cell selection and analyses."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QPushButton, QGroupBox, QScrollArea
from qtpy.QtCore import Qt
from gui.analysis.registry import AnalysisRegistry
from utils.logger import TA_logger

logger = TA_logger()


def create_analysis_tab(parent, delayed_preview_update):
    """Create and return the analysis tab widget with cell selection and analysis functionality.
    
    Args:
        parent: The main window instance (for connecting callbacks)
        delayed_preview_update: QTimer for delayed preview updates (kept for compatibility)
    
    Returns:
        QScrollArea: Scrollable container with the analysis tab
    """
    tab2_scroll = QScrollArea()
    tab2_scroll.setWidgetResizable(True)
    tab2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    tab2_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    tab2 = QWidget()
    tab2_scroll.setWidget(tab2)
    
    layout = QGridLayout()
    layout.setColumnStretch(0, 1)
    layout.setColumnStretch(1, 1)
    
    # Analyses group box (spans both columns)
    analyses_group = QGroupBox("Analyses")
    analyses_layout = QGridLayout()
    analyses_layout.setColumnStretch(0, 1)
    analyses_layout.setColumnStretch(1, 1)
    
    # Track completeness overlay button (spans both columns at the top)
    completeness_overlay_button = QPushButton("Show track completeness")
    completeness_overlay_button.setCheckable(True)
    completeness_overlay_button.setToolTip(
        "Enable/disable overlay showing track completeness.\n"
        "Green = complete tracks (present in all frames)\n"
        "Red = incomplete tracks (missing in some frames)")
    completeness_overlay_button.setChecked(False)
    completeness_overlay_button.clicked.connect(lambda checked: parent.toggle_completeness_overlay(checked))
    analyses_layout.addWidget(completeness_overlay_button, 0, 0, 1, 2)
    
    # Get registered analyses and create buttons
    # #region agent log
    try:
        with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
            f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"analysis_tab.py:create_analysis_tab:check_registry","message":"Checking for analysis_registry","data":{"has_registry":hasattr(parent,"analysis_registry")},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
    except: pass
    # #endregion
    if hasattr(parent, 'analysis_registry'):
        registry = parent.analysis_registry
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"analysis_tab.py:create_analysis_tab:get_analyses","message":"Getting all analyses","data":{},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
        except: pass
        # #endregion
        analysis_widgets = {}
        
        all_analyses = registry.get_all_analyses()
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                import json
                log_data = {"count": len(all_analyses), "names": all_analyses}
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"analysis_tab.py:create_analysis_tab:analyses_list","message":"Got analyses list","data":log_data,"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
        
        # Track actual analysis count for 2-column grid layout (start at row 1, since row 0 is the completeness button)
        analysis_count = 0
        for idx, analysis_name in enumerate(all_analyses):
            # #region agent log
            try:
                with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                    f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"analysis_tab.py:create_analysis_tab:loop_start","message":"Processing analysis","data":{"name":analysis_name},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
            except: pass
            # #endregion
            try:
                analysis = registry.get_analysis(analysis_name)
                # #region agent log
                try:
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"analysis_tab.py:create_analysis_tab:got_analysis","message":"Got analysis instance","data":{"name":analysis_name,"has_analysis":analysis is not None},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
                except: pass
                # #endregion
                if not analysis:
                    continue
                
                # Create a sub-box for this analysis
                analysis_sub_box = QGroupBox(analysis.name)
                analysis_sub_layout = QVBoxLayout()
                
                # Add "Choose cells" button for all analyses
                choose_cells_button = QPushButton("Choose cells")
                choose_cells_button.setToolTip(
                    f"Choose cells for {analysis.name}.\n"
                    f"{analysis.description}\n"
                    "Click to enable selection mode and select cells for this analysis.")
                
                # Fix lambda closure issue by capturing analysis_name in default parameter
                def make_choose_cells_handler(name):
                    def handler(checked):
                        parent.run_analysis(name)
                    return handler
                choose_cells_button.clicked.connect(make_choose_cells_handler(analysis_name))
                analysis_sub_layout.addWidget(choose_cells_button)
                
                # #region agent log
                try:
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"analysis_tab.py:create_analysis_tab:before_ui_widgets","message":"About to call get_ui_widgets","data":{"name":analysis_name},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
                except: pass
                # #endregion
                # Get custom UI widgets from analysis
                ui_widgets = analysis.get_ui_widgets()
                # #region agent log
                try:
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        import json
                        log_data = {"name": str(analysis_name), "widget_count": len(ui_widgets) if ui_widgets else 0}
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"analysis_tab.py:create_analysis_tab:after_ui_widgets","message":"Got UI widgets","data":log_data,"timestamp":int(__import__('time').time()*1000)}) + '\n')
                except: pass
                # #endregion
                
                if ui_widgets:
                    for widget_name, widget in ui_widgets.items():
                        if isinstance(widget, QWidget):
                            analysis_sub_layout.addWidget(widget)
                            if analysis_name not in analysis_widgets:
                                analysis_widgets[analysis_name] = {}
                            analysis_widgets[analysis_name][widget_name] = widget
                
                # Set layout and add sub-box to analyses layout in 2-column grid
                analysis_sub_box.setLayout(analysis_sub_layout)
                # Calculate column (0 or 1) and row based on actual analysis count
                # Row 0 is for the completeness button, so start at row 1
                col = analysis_count % 2
                row = 1 + (analysis_count // 2)
                analyses_layout.addWidget(analysis_sub_box, row, col)
                analysis_count += 1
                
                # #region agent log
                try:
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"analysis_tab.py:create_analysis_tab:loop_end","message":"Finished processing analysis","data":{"name":analysis_name},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
                except: pass
                # #endregion
            except Exception as e:
                # #region agent log
                try:
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        import traceback
                        import json
                        error_data = {"name": str(analysis_name), "error": str(e), "traceback": traceback.format_exc()}
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"analysis_tab.py:create_analysis_tab:exception","message":"Exception processing analysis","data":error_data,"timestamp":int(__import__('time').time()*1000)}) + '\n')
                except Exception as log_err:
                    pass
                # #endregion
                logger.error(f"Error processing analysis {analysis_name}: {e}")
                import traceback
                traceback.print_exc()
                raise  # Re-raise to see the actual exception
        
        # Store references on parent
        parent.analysis_widgets = analysis_widgets
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                import json
                log_data = {}
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"analysis_tab.py:create_analysis_tab:complete","message":"Analysis tab creation complete","data":log_data,"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
    else:
        logger.warning("Analysis registry not found on parent")
    
    analyses_group.setLayout(analyses_layout)
    
    # Add analyses group to layout spanning both columns
    layout.addWidget(analyses_group, 0, 0, 1, 2)
    
    tab2.setLayout(layout)
    
    # Store references on parent
    parent.tab2 = tab2
    parent.analysis_completeness_overlay_button = completeness_overlay_button
    
    return tab2_scroll
