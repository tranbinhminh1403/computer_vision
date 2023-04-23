import os
import numpy as np
import cv2

# print(os.listdir('image'))
# print(os.path.join('image', '1.jpg'))
# print(os.path.abspath('image'))
# print(os.path.isdir('image'))

#ex01
os.system('cls' if os.name == 'nt' else 'clear')
abs_path=[]
rel_path=[]

for file_name in os.listdir('image'):
    abs_file_path = os.path.abspath(os.path.join('image', file_name))
    abs_path.append(abs_file_path)
    rel_file_path = os.path.join('image', file_name)
    rel_path.append(rel_file_path)
print('absolute path:\n',abs_path)
print('relative path:\n',rel_path)
    
