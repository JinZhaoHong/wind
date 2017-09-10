from Preprocessor import Preprocessor
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

p = Preprocessor()
pos, X = p.preprocess()
lat = pos[:,0]
lon = pos[:,1]

fig = plt.figure(figsize = (8,8))
m = Basemap(projection='lcc', resolution = 'f',
              lat_0= (min(lat) + max(lat))/2.0, lon_0=(min(lon) + max(lon))/2.0,
              llcrnrlon=min(lon)-0.05, 
              llcrnrlat=min(lat)-0.05, 
              urcrnrlon=max(lon)+0.05, 
              urcrnrlat=max(lat)+0.05)
m.shadedrelief()
m.drawcoastlines(color='black')
m.drawcountries(color='gray')
m.drawstates(color='blue')

# 2. scatter city data, with color reflecting population
# and size reflecting area
print(min(lon)-0.05)
print(min(lat)-0.05)
print(max(lat)+0.05)
print(max(lon)+0.05)

x, y = m(lon, lat)
m.plot(x, y, 'ro')
plt.show()