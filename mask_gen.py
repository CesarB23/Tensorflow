import json
import cv2 
import numpy as np
import sys
sys.path

f = open("entrenamiento.json","r")
data = json.load(f)

imgs = r"C:\Users\52221\Desktop\Programacion\MaskRCNN\Brain\entrenamiento"

print(data)
