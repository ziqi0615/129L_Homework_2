#!/usr/bin/env python

from numpy import sin, cos, pi, linspace, meshgrid, full_like, gradient, stack, array, zeros_like
from matplotlib import pyplot as plt


def cartesian_coordinates(theta_angle, phi_angle):
	return (sin(theta_angle)*cos(phi_angle), sin(theta_angle)*sin(phi_angle), cos(theta_angle))

def unit_vector_theta(theta_angle, phi_angle):
	return cartesian_coordinates(theta_angle+pi/2, phi_angle)

def unit_vector_phi(theta_angle, phi_angle):
	return (-sin(theta_angle)*sin(phi_angle), sin(theta_angle)*cos(phi_angle), zeros_like(theta_angle))

def local_coordinates(f, x_values, y_values):
	dx_values = x_values[1:,:] - x_values[:-1,:]
	dy_values = y_values[:,1:] - y_values[:,:-1]
	z_values = f(x_values, y_values)
	dfdx_values = gradient(z_values, dx_values, axis=0)
	dfdy_values = gradient(z_values, dy_values, axis=1)
	norm_values = (dfdx_values**2 + dfdy_values**2 + 1)**0.5
	return stack((dfdx_values/norm_values, dfdy_values/norm_values, -1/norm_values), axis=-1)

theta_angle = linspace(0, pi, 20)
phi_angle = linspace(0, 2*pi, 40)
theta_angle, phi_angle = meshgrid(theta_angle, phi_angle)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(*cartesian_coordinates(theta_angle, phi_angle))

phi_angle = linspace(0, 2*pi, 20)
theta_angle = full_like(phi_angle, pi/4)
x0_theta = 0.1
x0_phi = 0.1
x_theta = x0_theta*cos(phi_angle*cos(theta_angle)) + x0_phi*sin(theta_angle)*sin(phi_angle*cos(theta_angle))
x_phi = x0_phi*cos(phi_angle*cos(theta_angle)) - x0_theta/sin(theta_angle)*sin(phi_angle*cos(theta_angle))
ax.quiver(*cartesian_coordinates(theta_angle, phi_angle), *(x_theta*array(unit_vector_theta(theta_angle,phi_angle))+x_phi*array(unit_vector_phi(theta_angle,phi_angle))), color='r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()

# g
theta_angle = linspace(0, pi, 200)
strength = cos(2*pi*cos(theta_angle))
plt.plot(theta_angle, strength)
plt.xlabel(r'$\theta_0$')
plt.ylabel('strength')
plt.show()
