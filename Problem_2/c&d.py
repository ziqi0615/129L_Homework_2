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

def stereo(theta, phi):
	return sin(theta)*cos(phi)/(1-cos(theta)), sin(theta)*sin(phi)/(1-cos(theta))

def stereo_vector(theta, phi, x_theta, x_phi):
	x_x = (-cos(phi)*x_theta - sin(theta)*sin(phi)*x_phi) / (1-cos(theta))
	x_y = (-sin(phi)*x_theta + sin(theta)*cos(phi)*x_phi) / (1-cos(theta))
	return x_x, x_y

def transport(theta, phi, x0_theta, x0_phi):
	x_theta = x0_theta*cos(phi*cos(theta)) + x0_phi*sin(theta)*sin(phi*cos(theta))
	x_phi = x0_phi*cos(phi*cos(theta)) - x0_theta/sin(theta)*sin(phi*cos(theta))
	return x_theta, x_phi

phi = linspace(0, 2*pi, 20)
theta = full_like(phi, pi/4)
plt.plot(*stereo(theta, phi), color='b')
plt.quiver(*stereo(theta, phi), *stereo_vector(theta, phi, *transport(theta, phi, 0.2, 0.1)), color='r')
plt.quiver(*stereo(theta, phi), *stereo_vector(theta, phi, *transport(theta, phi, 0.1, 0.2)), color='g')
plt.xlabel("$x'$")
plt.ylabel("$y'$")
plt.show()

phi = linspace(0, 2*pi, 200)
theta = full_like(phi, pi/4)
x1_x, x1_y = stereo_vector(theta, phi, *transport(theta, phi, 0.2, 0.1))
x2_x, x2_y = stereo_vector(theta, phi, *transport(theta, phi, 0.1, 0.2))
plt.plot(phi, (x1_x*x2_x + x1_y*x2_y) * 4/(1+x1_x**2+x1_y**2)**2)
plt.xlabel("$\\phi$")
plt.ylabel("Inner product")
plt.show()
