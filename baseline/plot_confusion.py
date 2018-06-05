import numpy as np
import matplotlib.pyplot as plt

conf_arr = [[0.3764, 0.1613],[0.1156, 0.3467]]

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(conf_arr), cmap=plt.cm.jet, interpolation='nearest')

width, height = conf_arr.shape

for x in xrange(width):
	for y in xrange(height):
		ax.annotate(str(conf_arr[x][y]), xy=(y,x),
				horizontalalignment='center',
				verticalalignment='center')

cb = fig.colorbar(res)
plt.savefig('confusion_matrixx.png', format='png')
