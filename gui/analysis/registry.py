"""Analysis registry for discovering and managing analyses."""

import os
import importlib
import inspect
from typing import Any, Dict, List, Optional, Type
from gui.analysis.base_analysis import BaseAnalysis
from utils.logger import TA_logger

logger = TA_logger()


class AnalysisRegistry:
    """Registry for managing available analyses."""
    
    def __init__(self):
        self._analyses: Dict[str, Type[BaseAnalysis]] = {}
        self._instances: Dict[str, BaseAnalysis] = {}
        self._load_analyses()
    
    def _load_analyses(self):
        """Auto-discover and load analyses from the analyses directory."""
        analyses_dir = os.path.join(os.path.dirname(__file__), 'analyses')
        
        if not os.path.exists(analyses_dir):
            logger.warning(f"Analyses directory not found: {analyses_dir}")
            return
        
        # Try to import from analyses package
        try:
            import importlib
            analyses_module = importlib.import_module('gui.analysis.analyses')
            # #region agent log
            try:
                import json, time
                with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                    module_attrs = [name for name in dir(analyses_module) if not name.startswith('_')]
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"registry.py:_load_analyses:module_loaded","message":"Analyses module loaded","data":{"module_attrs":module_attrs,"__all__":getattr(analyses_module, '__all__', [])},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            # Get all classes from the analyses module
            for name in dir(analyses_module):
                if name.startswith('_'):
                    continue
                obj = getattr(analyses_module, name)
                # #region agent log
                try:
                    import json, time
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        is_class = inspect.isclass(obj)
                        is_subclass = False
                        is_base = False
                        if is_class:
                            try:
                                is_subclass = issubclass(obj, BaseAnalysis)
                                is_base = (obj is BaseAnalysis)
                            except:
                                pass
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"registry.py:_load_analyses:checking_attr","message":"Checking attribute","data":{"name":name,"is_class":is_class,"is_subclass":is_subclass,"is_base":is_base},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseAnalysis) and 
                    obj is not BaseAnalysis):
                    self.register(obj)
        except ImportError as e:
            logger.warning(f"Could not import analyses module: {e}")
            # #region agent log
            try:
                import json, time, traceback
                with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"registry.py:_load_analyses:import_error","message":"Import error","data":{"error":str(e),"traceback":traceback.format_exc()},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
        except Exception as e:
            logger.error(f"Error loading analyses: {e}")
            import traceback
            traceback.print_exc()
            # #region agent log
            try:
                import json, time
                with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"registry.py:_load_analyses:exception","message":"Exception loading analyses","data":{"error":str(e),"traceback":traceback.format_exc()},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
    
    def register(self, analysis_class: Type[BaseAnalysis]):
        """
        Register an analysis class.
        
        Args:
            analysis_class: Class that extends BaseAnalysis
        """
        if not issubclass(analysis_class, BaseAnalysis):
            raise ValueError(f"{analysis_class} must extend BaseAnalysis")
        
        # Create instance to get name
        try:
            instance = analysis_class()
            name = instance.name
            if not name:
                name = analysis_class.__name__
            
            self._analyses[name] = analysis_class
            logger.debug(f"Registered analysis: {name}")
        except Exception as e:
            logger.error(f"Error registering analysis {analysis_class}: {e}")
    
    def get_analysis(self, name: str) -> Optional[BaseAnalysis]:
        """
        Get an analysis instance by name.
        
        Args:
            name: Analysis name
        
        Returns:
            Analysis instance or None if not found
        """
        if name not in self._instances:
            if name in self._analyses:
                try:
                    self._instances[name] = self._analyses[name]()
                except Exception as e:
                    logger.error(f"Error creating analysis instance {name}: {e}")
                    return None
            else:
                return None
        
        return self._instances.get(name)
    
    def get_all_analyses(self) -> List[str]:
        """Get list of all registered analysis names."""
        return list(self._analyses.keys())
    
    def get_analysis_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an analysis.
        
        Args:
            name: Analysis name
        
        Returns:
            Dictionary with analysis info or None if not found
        """
        analysis = self.get_analysis(name)
        if not analysis:
            return None
        
        return {
            'name': analysis.name,
            'description': analysis.description,
            'selection_mode': analysis.selection_mode,
            'selection_count': analysis.selection_count,
        }
