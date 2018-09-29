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

from osgeo import gdal
gtif = gdal.Open( "testando.tif" )
print(gtif)