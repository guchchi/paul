from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseVisionModel(ABC):
    """
    Base interface for all Vision Models to ensure plug-and-play compatibility 
    between Prototype CV Heuristics and future Deep Learning Backends.
    """
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Returns the type of backend (e.g. 'Prototype Vision Heuristic' or 'Deep Learning U-Net')"""
        pass
        
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs the model inference on the image.
        Returns a dictionary of structured results.
        """
        pass

class BaseAnomalyModel(ABC):
    """
    Base interface for visual anomaly detection models.
    """
    
    @abstractmethod
    def get_backend_name(self) -> str:
        pass
        
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Returns anomaly heatmap and scores.
        """
        pass
