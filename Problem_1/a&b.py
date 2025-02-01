#!/usr/bin/env python

from numpy import sin, cos, pi, linspace, meshgrid, full_like, gradient, stack, array, zeros_like
from matplotlib import pyplot as plt

def spherical_coordinates(angle_theta, angle_phi):
	return (full_like(angle_theta, 1), angle_theta, angle_phi)

def cartesian_coordinates(angle_theta, angle_phi):
	return (sin(angle_theta)*cos(angle_phi), sin(angle_theta)*sin(angle_phi), cos(angle_theta))

def cylindrical_coordinates(angle_theta, angle_phi):
	return (sin(angle_theta), angle_phi, cos(angle_theta))


def unit_vector_r(angle_theta, angle_phi):
	return cartesian_coordinates(angle_theta, angle_phi)
def unit_vector_theta(angle_theta, angle_phi):
	return cartesian_coordinates(angle_theta+pi/2, angle_phi)
def unit_vector_phi(angle_theta, angle_phi):
	return (-sin(angle_theta)*sin(angle_phi), sin(angle_theta)*cos(angle_phi), zeros_like(angle_theta))

angle_theta = linspace(0, pi/2, 5)
angle_phi = linspace(0, 2*pi, 20)
angle_theta, angle_phi = meshgrid(angle_theta, angle_phi)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_surface(*cartesian_coordinates(angle_theta, angle_phi))
ax.quiver(*cartesian_coordinates(angle_theta, angle_phi), *unit_vector_r(angle_theta, angle_phi), length=0.1, color='r')
ax.quiver(*cartesian_coordinates(angle_theta, angle_phi), *unit_vector_theta(angle_theta, angle_phi), length=0.1, color='g')
ax.quiver(*cartesian_coordinates(angle_theta, angle_phi), *unit_vector_phi(angle_theta, angle_phi), length=0.1, color='b')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()
