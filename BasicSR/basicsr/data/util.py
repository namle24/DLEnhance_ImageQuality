import cv2
import numpy as np
import torch
from pathlib import Path

def imfrombytes(content, flag='color'):
    img_np = np.frombuffer(content, np.uint8)
    flag = cv2.IMREAD_COLOR if flag == 'color' else cv2.IMREAD_GRAYSCALE
    img = cv2.imdecode(img_np, flag)
    return img

class FileClient:
    def __init__(self):
        pass
    
    def get(self, filepath):
        with open(filepath, 'rb') as f:
            content = f.read()
        return content