import cv2
import numpy as np

class ImagePreprocessor:
    """
    Standardizes inputs for the vision pipeline.
    """
    
    def process(self, image: np.ndarray, target_width: int = 800) -> np.ndarray:
        """
        Resizes the image and applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        """
        # Resize maintaining aspect ratio
        h, w = image.shape[:2]
        if w > target_width:
            ratio = target_width / w
            image = cv2.resize(image, (target_width, int(h * ratio)))
            
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) is removed 
        # because it artificially creates high-contrast lines in natural textures 
        # (grass, dirt roads), confusing the YOLO and CV crack models.
        
        # Mild denoising on the raw resized image
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        
        return denoised
