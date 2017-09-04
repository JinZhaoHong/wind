from Preprocessor import Preprocessor
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

p = Preprocessor()
pos, X = p.preprocess()

lat = pos[:,0]
lon = pos[:,1]
x = np.linspace(min(lon) - 0.03, max(lon) + 0.03)
y = np.linspace(min(lat) - 0.03, max(lat) + 0.03)
for i in range(50) :
	print(i % 24)
	plt.figure()
	capacity = X[:,i]
	z = griddata((lon, lat), capacity, (x[None,:], y[:,None]))
	cs = plt.contour(x, y, z)
	plt.plot(lon, lat, 'ro')
	plt.clabel(cs, cs.levels, inline=True, fontsize=10)
	
	for label, xi, yi in zip(capacity, lon, lat) :
		plt.annotate(
			np.int(label),
			xy = (xi, yi), xytext=(5,0),
			textcoords = 'offset points')
	plt.show()