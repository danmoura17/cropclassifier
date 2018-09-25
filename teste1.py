# from osgeo import gdal
# import matplotlib.pyplot as plt

# ds = gdal.Open('sugar1.tif').ReadAsArray()

# im = plt.imshow(ds)

# im.savefig('output.tif', dpi=im.dpi)

# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(3, 6))

# plt.plot(range(10)) #plot example
# plt.show() #for control

# fig.savefig('temp.png', dpi=fig.dpi)

import matplotlib
import matplotlib.pyplot as plt

plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])

plt.xlabel('Months')
plt.ylabel('Books Read')
plt.show()

plt.savefig('books_read.png')