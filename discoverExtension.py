import os

countjpg = 0

for root, dirs, files in os.walk("."):  
	for filename in files:
		countjpg += 1
		os.rename(filename, ('other' + str(countjpg) + '.tif'))

