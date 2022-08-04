import numpy as np
from matplotlib import pyplot as plt
from .utils import print_arr

# Subdivision steps
r = 9

#from trimesh.creation import icosahedron
#ico = icosahedron()
#num_sides = ico.triangles.shape[0]
#for _ in range(r):
#    ico = ico.subdivide()

from trimesh.creation import icosphere
ico = icosphere(r)

num_tris = ico.triangles.shape[0]
num_sides = num_tris
print(f'{num_tris} triangles, {num_sides} sides, {num_tris//num_sides} triangles per side')

#normals = ico.face_normals
#coords = ico.triangles_center
coords = ico.vertices
h = np.linalg.norm(coords, axis=1)

pano = plt.imread('pano.jpg')
num_y, num_x, num_c = pano.shape

print(f'{num_x} x {num_y} = {num_x*num_y} pixels')
print(f'{num_x*num_y/num_tris:.2f} pixels per triangle')

#x_long = np.linspace(-np.pi,   +np.pi,   num_x)
#y_lat  = np.linspace(-np.pi/2, +np.pi/2, num_y)

longs = np.arctan2(coords[:, 1]/h, coords[:, 0]/h)
lats  = np.arcsin(coords[:, 2]/h)

#print((coords/h[:, None]).max(axis=0))
print_arr('longs', longs)
print_arr('lats', lats)

#ico.visual.face_colors[:, 0:3] = 0
#ico.visual.face_colors[:, 0] = (+255*longs/2/np.pi + 127).astype(np.uint8)
#ico.visual.face_colors[:, 1] = (-255*longs/2/np.pi + 127).astype(np.uint8)

longs_x = ((longs + np.pi) / np.pi / 2 * (num_x - 1)).astype(np.int32)
lats_y  = ((-lats + np.pi/2) / (np.pi/2) / 2 * (num_y - 1)).astype(np.int32)

print_arr('longs_x', longs_x)
print_arr('lats_y', lats_y)

ico.visual.vertex_colors[:, 0:num_c] = pano[lats_y, longs_x, :]

print_arr('face_colors', ico.visual.face_colors)

ico.show()

#import code ; code.interact(local=locals())
