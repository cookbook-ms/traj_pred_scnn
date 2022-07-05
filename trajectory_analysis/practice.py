
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay, delaunay_plot_2d
import os
import matplotlib.pyplot as plt
import jax.numpy as jnp 
from jax.scipy.special import logsumexp

np.random.seed(1)
n = 10
points = np.random.rand(n,2)

tri = Delaunay(points)
print(tri.simplices)
print(points[0,:])
print(points[tri.simplices])

_ = delaunay_plot_2d(tri)
#plt.show()

# plt.triplot(points[:,0], points[:,1], tri.simplices)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()

logsumexp(np.array[1,2,8])