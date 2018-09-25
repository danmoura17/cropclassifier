import mimetypes
import magic
import os

countjpg = 0
countpng = 0

for root, dirs, files in os.walk("."):  
	for filename in files:
		fileType = magic.from_file(filename, mime = True)
		if (fileType == 'image/jpeg'):
			countjpg += 1
			os.rename(filename, ('other' + str(countjpg) + '.jpeg'))
		elif (fileType == 'image/png'):
			countpng += 1
			os.rename(filename, ('other' + str(countpng) + '.png'))
		else:
			print(fileType)
			print(filename)
			print('------')
