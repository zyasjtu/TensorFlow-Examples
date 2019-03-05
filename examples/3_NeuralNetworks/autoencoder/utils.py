import cv2
import numpy as np
import json

def load_image(fn, w, h, cvt2gray):
	img = cv2.imread(fn)
	img = cv2.resize(img, (w, h))
        if (cvt2gray):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	return np.array(img)

def load_data(data_path):
	f = open(data_path)
	path_class_list = f.readlines()
	f.close()
	file_path = []
	for i in range(len(path_class_list)):
		file_path.append(path_class_list[i])
	return file_path

def get_cfg(cfg_path):
	with open(cfg_path) as json_file:
		data = json.load(json_file)
		return data
