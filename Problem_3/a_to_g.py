#!/usr/bin/env python

from numpy import array, unique, argmin, argmax, cross, loadtxt, stack, vstack, pad, full_like, dot
from numpy.linalg import inv, eig
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay


def quickhull(vertices):
    def add_hull(vertices, point1, point2):
        if not len(vertices):
            return []
        distances = cross(vertices - point1, point2 - point1)
        farthest_point_index = argmax(distances)
        farthest_point = vertices[farthest_point_index]
        left_set_1 = vertices[cross(vertices - point1, farthest_point - point1) > 0]
        left_set_2 = vertices[cross(vertices - farthest_point, point2 - farthest_point) > 0]
        return add_hull(left_set_1, point1, farthest_point) + [farthest_point] + add_hull(left_set_2, farthest_point, point2)

    vertices = unique(vertices, axis=0)
    if len(vertices) < 3:
        return vertices
    leftmost_vertex = vertices[argmin(vertices[:, 0])]
    rightmost_vertex = vertices[argmax(vertices[:, 0])]
    upper_set = vertices[cross(vertices - leftmost_vertex, rightmost_vertex - leftmost_vertex) > 0]
    lower_set = vertices[cross(vertices - leftmost_vertex, rightmost_vertex - leftmost_vertex) < 0]
    upper_hull = add_hull(upper_set, leftmost_vertex, rightmost_vertex)
    lower_hull = add_hull(lower_set, rightmost_vertex, leftmost_vertex)
    return array([leftmost_vertex] + upper_hull + [rightmost_vertex] + lower_hull)


def triangle_area(vertex1, vertex2, vertex3):
    vertex1 = pad(vertex1, (0, 3-len(vertex1)), 'constant')
    vertex2 = pad(vertex2, (0, 3-len(vertex2)), 'constant')
    vertex3 = pad(vertex3, (0, 3-len(vertex3)), 'constant')
    return ((cross(vertex2 - vertex1, vertex3 - vertex1) / 2) ** 2).sum() ** 0.5


data_points = loadtxt('/root/Desktop/host/section_2_129L/mesh.dat', skiprows=1)
data_points = data_points * 0.4 - 2  # Scale and translate points
convex_hull = quickhull(data_points)
convex_hull = vstack((convex_hull, convex_hull[0]))
plt.plot(convex_hull[:, 0], convex_hull[:, 1], color='r')

triangular_mesh = Delaunay(data_points).simplices
plt.triplot(data_points[:, 0], data_points[:, 1], triangular_mesh, color='g')

plt.scatter(data_points[:, 0], data_points[:, 1], color='b')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()


def surface_function(x_coord, y_coord):
    return x_coord ** 2 + y_coord ** 2


face_colors = []
for triangle in triangular_mesh:
    planar_area = triangle_area(*data_points[triangle])
    lifted_area = triangle_area(*[[*point, surface_function(*point)] for point in data_points[triangle]])
    face_colors.append(planar_area / lifted_area)

tri_color = plt.tripcolor(data_points[:, 0], data_points[:, 1], facecolors=face_colors, triangles=triangular_mesh, cmap='viridis')
plt.colorbar(tri_color)
plt.scatter(data_points[:, 0], data_points[:, 1], color='b')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()


def triangle_normal(vertex1, vertex2, vertex3):
    normal_vector = cross(vertex3 - vertex1, vertex2 - vertex1)
    return normal_vector / (normal_vector ** 2).sum() ** 0.5


def centroid(vertex1, vertex2, vertex3):
    return (vertex1 + vertex2 + vertex3) / 3


points_3d = vstack((data_points.T, surface_function(*data_points.T))).T
centroids = array([centroid(*points_3d[triangle]) for triangle in triangular_mesh])
normals = array([triangle_normal(*points_3d[triangle]) for triangle in triangular_mesh])

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(*points_3d.T, triangles=triangular_mesh, cmap='viridis')
ax.quiver(*centroids.T, *normals.T, color='r', length=0.5)
plt.show()

vertex_normals = []
for idx in range(len(data_points)):
    triangles_with_vertex = [tri for tri in triangular_mesh if idx in tri]
    normal_vector = sum(triangle_normal(*points_3d[tri]) * triangle_area(*points_3d[tri]) for tri in triangles_with_vertex)
    vertex_normals.append(normal_vector / (normal_vector ** 2).sum() ** 0.5)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(*points_3d.T, triangles=triangular_mesh, cmap='viridis')
ax.quiver(*points_3d.T, *array(vertex_normals).T, color='r', length=0.5)
plt.show()

x_coords = points_3d[:, 0]
y_coords = points_3d[:, 1]
partial_deriv_x = array([full_like(x_coords, 1), full_like(x_coords, 0), 2 * x_coords]).T
partial_deriv_y = array([full_like(y_coords, 0), full_like(y_coords, 1), 2 * y_coords]).T
partial_xx = array([full_like(x_coords, 0), full_like(x_coords, 0), full_like(x_coords, 2)]).T
partial_yy = array([full_like(y_coords, 0), full_like(y_coords, 0), full_like(y_coords, 2)]).T
partial_xy = array([full_like(x_coords, 0), full_like(x_coords, 0), full_like(x_coords, 0)]).T

second_fundamental_xx = (partial_xx * vertex_normals).sum(axis=1)
second_fundamental_yy = (partial_yy * vertex_normals).sum(axis=1)
second_fundamental_xy = (partial_xy * vertex_normals).sum(axis=1)
second_fundamental_yx = second_fundamental_xy
second_fundamental = stack([second_fundamental_xx, second_fundamental_xy, second_fundamental_yx, second_fundamental_yy], axis=-1).reshape(-1, 2, 2)

first_fundamental_xx = (partial_deriv_x * partial_deriv_x).sum(axis=1)
first_fundamental_yy = (partial_deriv_y * partial_deriv_y).sum(axis=1)
first_fundamental_xy = (partial_deriv_x * partial_deriv_y).sum(axis=1)
first_fundamental_yx = first_fundamental_xy
first_fundamental = inv(stack([first_fundamental_xx, first_fundamental_xy, first_fundamental_yx, first_fundamental_yy], axis=-1).reshape(-1, 2, 2))

shape_operator = array([dot(first_fundamental[i], second_fundamental[i]) for i in range(len(data_points))])

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
