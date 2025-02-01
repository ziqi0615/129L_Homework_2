#!/usr/bin/env python

from numpy import array, unique, argmin, argmax, cross, loadtxt, stack, vstack, pad, full_like, dot
from numpy.linalg import inv, eig
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

# a
def compute_hull(vertices):
    def add_to_hull(vertices, vertex1, vertex2):
        if not len(vertices):
            return []
        distance_values = cross(vertices - vertex1, vertex2 - vertex1)
        max_distance_index = argmax(distance_values)
        farthest_vertex = vertices[max_distance_index]
        left_subset_1 = vertices[cross(vertices - vertex1, farthest_vertex - vertex1) > 0]
        left_subset_2 = vertices[cross(vertices - farthest_vertex, vertex2 - farthest_vertex) > 0]
        return add_to_hull(left_subset_1, vertex1, farthest_vertex) + [farthest_vertex] + add_to_hull(left_subset_2, farthest_vertex, vertex2)

    vertices = unique(vertices, axis=0)
    if len(vertices) < 3:
        return vertices
    leftmost_vertex = vertices[argmin(vertices[:, 0])]
    rightmost_vertex = vertices[argmax(vertices[:, 0])]
    upper_subset = vertices[cross(vertices - leftmost_vertex, rightmost_vertex - leftmost_vertex) > 0]
    lower_subset = vertices[cross(vertices - leftmost_vertex, rightmost_vertex - leftmost_vertex) < 0]
    upper_hull = add_to_hull(upper_subset, leftmost_vertex, rightmost_vertex)
    lower_hull = add_to_hull(lower_subset, rightmost_vertex, leftmost_vertex)
    return array([leftmost_vertex] + upper_hull + [rightmost_vertex] + lower_hull)

def calculate_triangle_area(vertex1, vertex2, vertex3):
    vertex1 = pad(vertex1, (0, 3 - len(vertex1)), 'constant')
    vertex2 = pad(vertex2, (0, 3 - len(vertex2)), 'constant')
    vertex3 = pad(vertex3, (0, 3 - len(vertex3)), 'constant')
    return ((cross(vertex2 - vertex1, vertex3 - vertex1) / 2) ** 2).sum() ** 0.5

data_vertices = loadtxt('/root/Desktop/host/section_2_129L/mesh.dat', skiprows=1)
data_vertices = data_vertices * 0.4 - 2 
convex_hull = compute_hull(data_vertices)
convex_hull = vstack((convex_hull, convex_hull[0]))
plt.plot(convex_hull[:, 0], convex_hull[:, 1], color='r')

triangle_indices = Delaunay(data_vertices).simplices
plt.triplot(data_vertices[:, 0], data_vertices[:, 1], triangle_indices, color='g')

plt.scatter(data_vertices[:, 0], data_vertices[:, 1], color='b')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

# b
def compute_surface(x_coord, y_coord):
    return x_coord**2 + x_coord * y_coord + y_coord**2

face_colors = []
for triangle in triangle_indices:
    planar_area = calculate_triangle_area(*data_vertices[triangle])
    elevated_area = calculate_triangle_area(*[[*point, compute_surface(*point)] for point in data_vertices[triangle]])
    face_colors.append(planar_area / elevated_area)

tri_color = plt.tripcolor(data_vertices[:, 0], data_vertices[:, 1], facecolors=face_colors, triangles=triangle_indices, cmap='viridis')
plt.colorbar(tri_color)
plt.scatter(data_vertices[:, 0], data_vertices[:, 1], color='b')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

# c
# g_xx = 1 + (2x + y)^2
# g_yy = 1 + (x + 2y)^2
# g_xy = g_yx = (2x + y)(x + 2y)

# d
def compute_normal(vertex1, vertex2, vertex3):
    normal_vector = cross(vertex3 - vertex1, vertex2 - vertex1)
    return normal_vector / (normal_vector ** 2).sum() ** 0.5

def compute_centroid(vertex1, vertex2, vertex3):
    return (vertex1 + vertex2 + vertex3) / 3

three_dimensional_vertices = vstack((data_vertices.T, compute_surface(*data_vertices.T))).T
triangle_centroids = array([compute_centroid(*three_dimensional_vertices[triangle]) for triangle in triangle_indices])
triangle_normals = array([compute_normal(*three_dimensional_vertices[triangle]) for triangle in triangle_indices])

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(*three_dimensional_vertices.T, triangles=triangle_indices, cmap='viridis')
ax.quiver(*triangle_centroids.T, *triangle_normals.T, color='r', length=0.5)
plt.show()

# e
vertex_normals = []
for index in range(len(data_vertices)):
    triangles_with_vertex = [triangle for triangle in triangle_indices if index in triangle]
    averaged_normal = sum(compute_normal(*three_dimensional_vertices[triangle]) * calculate_triangle_area(*three_dimensional_vertices[triangle]) for triangle in triangles_with_vertex)
    vertex_normals.append(averaged_normal / (averaged_normal ** 2).sum() ** 0.5)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(*three_dimensional_vertices.T, triangles=triangle_indices, cmap='viridis')
ax.quiver(*three_dimensional_vertices.T, *array(vertex_normals).T, color='r', length=0.5)
plt.show()

# f
x_coords = three_dimensional_vertices[:, 0]
y_coords = three_dimensional_vertices[:, 1]
partial_derivative_x = array([full_like(x_coords, 1), full_like(x_coords, 0), 2 * x_coords + y_coords]).T
partial_derivative_y = array([full_like(y_coords, 0), full_like(y_coords, 1), 2 * y_coords + x_coords]).T
partial_derivative_xx = array([full_like(x_coords, 0), full_like(x_coords, 0), full_like(x_coords, 2)]).T
partial_derivative_yy = array([full_like(y_coords, 0), full_like(y_coords, 0), full_like(y_coords, 2)]).T
partial_derivative_xy = array([full_like(x_coords, 0), full_like(x_coords, 0), full_like(x_coords, 1)]).T

second_fundamental_xx = (partial_derivative_xx * vertex_normals).sum(axis=1)
second_fundamental_yy = (partial_derivative_yy * vertex_normals).sum(axis=1)
second_fundamental_xy = (partial_derivative_xy * vertex_normals).sum(axis=1)
second_fundamental_yx = second_fundamental_xy
second_fundamental = stack([second_fundamental_xx, second_fundamental_xy, second_fundamental_yx, second_fundamental_yy], axis=-1).reshape(-1, 2, 2)

# g
first_fundamental_xx = (partial_derivative_x * partial_derivative_x).sum(axis=1)
first_fundamental_yy = (partial_derivative_y * partial_derivative_y).sum(axis=1)
first_fundamental_xy = (partial_derivative_x * partial_derivative_y).sum(axis=1)
first_fundamental_yx = first_fundamental_xy
first_fundamental = inv(stack([first_fundamental_xx, first_fundamental_xy, first_fundamental_yx, first_fundamental_yy], axis=-1).reshape(-1, 2, 2))

shape_operator = array([dot(first_fundamental[i], second_fundamental[i]) for i in range(len(data_vertices))])

principal_curvatures = eig(shape_operator).eigenvalues
gaussian_curvature = principal_curvatures.prod(axis=-1)
mean_curvature = principal_curvatures.sum(axis=-1)

plt.scatter(x_coords, y_coords, c=gaussian_curvature, cmap='viridis')
plt.colorbar(label='Gaussian curvature')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

plt.scatter(x_coords, y_coords, c=mean_curvature, cmap='viridis')
plt.colorbar(label='Mean curvature')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
