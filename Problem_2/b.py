#!/usr/bin/env python

from numpy import exp, log, sin, cos, arccos, pi, linspace, meshgrid, full_like, array, cross, outer
from matplotlib import pyplot as plt

def cartesian_coordinates(theta_angle, phi_angle):
	return (sin(theta_angle)*cos(phi_angle), sin(theta_angle)*sin(phi_angle), cos(theta_angle))

theta_angle = linspace(0, pi, 20)
phi_angle = linspace(0, 2*pi, 40)
theta_angle, phi_angle = meshgrid(theta_angle, phi_angle)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(*cartesian_coordinates(theta_angle, phi_angle), color='grey')

def great_circle(normal_vec):
	normal_vec /= (array(normal_vec, dtype='float64')**2).sum()**0.5
	p1 = array([-normal_vec[1], normal_vec[0], 0])
	p1 /= (p1**2).sum()**0.5
	p2 = cross(normal_vec, p1)
	psi = linspace(0, 2*pi, 100)
	return outer(p1, cos(psi)) + outer(p2, sin(psi))

ax.plot(*great_circle((1,2,3)), color='r')
ax.plot(*great_circle((1,1,-1)), color='g')
ax.plot(*great_circle((5,-2,8)), color='b')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()

def stereo_circle(normal_vec):
	x, y, z = great_circle(normal_vec)
	return x/(1-z), y/(1-z)

plt.plot(*stereo_circle((1,2,3)), color='r')
plt.plot(*stereo_circle((1,1,-1)), color='g')
plt.plot(*stereo_circle((5,-2,8)), color='b')
plt.xlabel("$x'$")
plt.ylabel("$y'$")
plt.show()