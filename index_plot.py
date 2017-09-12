from Preprocessor import Preprocessor
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

p = Preprocessor()
pos, X = p.preprocess(pca = True)

lat = pos[:,0]
lon = pos[:,1]

x = np.linspace(min(lon) - 0.03, max(lon) + 0.03)
y = np.linspace(min(lat) - 0.03, max(lat) + 0.03)


label = np.asarray(range(75))
print(label)
print(pos.shape)
z = griddata((lon, lat), label, (x[None, :], y[:, None]))
cs = plt.contour(x, y, z)
plt.plot(lon, lat, "ro")
plt.clabel(cs, cs.levels, inline = True, fontsize = 10)
for l, xi, yi, in zip(label, lon, lat) :
	plt.annotate(
		l,
		xy = (xi,yi), xytext = (5,0),
		textcoords = 'offset points')
plt.show()