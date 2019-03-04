from PIL import Image
from os import listdir
from os.path import splitext

target_directory = 'C:\\Users\\Ben\\Desktop\\data\\nonad'
target = '.jpg'

counter = 0

for file in listdir(target_directory):
	if counter >= 5:
		break
	filename, extension = splitext(file)
	print("Filename : {}, extension: {}".format(filename, extension))
	try:
	    im = Image.open(target_directory + filename + extension)
	    im.save(filename + target, "JPEG")
	except OSError as e:
	    print(e)
	counter += 1