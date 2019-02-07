import numpy as np
import cv2
import json
import sys

webpage = sys.argv[1]
img_file = "{}/fullpage.png".format(webpage)
img = cv2.imread(img_file, -1)

frames_file = "{}/frames.txt".format(webpage)
with open(frames_file) as f:
	frames = json.load(f)
	
	for i, frame in enumerate(frames):
		x0, y0, w, h = [int(_) for _ in frame]
		if w*h > 0:
			frame_img = img[y0:y0+h, x0:x0+w, :]

			frame_img = cv2.resize(frame_img, None, fx=0.5, fy=0.5)

			cv2.imwrite("{}/frame_{}.png".format(webpage, i), frame_img)
