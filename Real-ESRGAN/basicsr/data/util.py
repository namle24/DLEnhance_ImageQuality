import cv2
import numpy as np
import torch
from pathlib import Path

def imfrombytes(content, flag='color'):
    img_np = np.frombuffer(content, np.uint8)
    if flag == 'color':
        flag = cv2.IMREAD_COLOR
    elif flag == 'grayscale':
        flag = cv2.IMREAD_GRAYSCALE
    elif flag == 'unchanged':
        flag = cv2.IMREAD_UNCHANGED
    img = cv2.imdecode(img_np, flag)
    if float32:
        img = img.astype(np.float32) / 255.
    return img

class FileClient:
    def __init__(self):
        pass
    
    def get(self, filepath):
        with open(filepath, 'rb') as f:
            content = f.read()
        return content