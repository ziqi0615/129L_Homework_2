#!/usr/bin/env python

from numpy import sin, cos, pi, linspace, meshgrid, full_like, gradient, stack, array, zeros_like
from matplotlib import pyplot as plt

def spherical_coordinates(theta_angle, phi_angle):
	return (full_like(theta_angle, 1), theta_angle, phi_angle)

def cartesian_coordinates(theta_angle, phi_angle):
	return (sin(theta_angle)*cos(phi_angle), sin(theta_angle)*sin(phi_angle), cos(theta_angle))

def cylindrical_coordinates(theta_angle, phi_angle):
	return (sin(theta_angle), phi_angle, cos(theta_angle))


def unit_vector_r(theta_angle, phi_angle):
	return cartesian_coordinates(theta_angle, phi_angle)
def unit_vector_theta(theta_angle, phi_angle):
	return cartesian_coordinates(theta_angle+pi/2, phi_angle)
def unit_vector_phi(theta_angle, phi_angle):
	return (-sin(theta_angle)*sin(phi_angle), sin(theta_angle)*cos(phi_angle), zeros_like(theta_angle))

theta_angle = linspace(0, pi/2, 5)
phi_angle = linspace(0, 2*pi, 20)
theta_angle, phi_angle = meshgrid(theta_angle, phi_angle)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_surface(*spherical_coordinates(theta_angle, phi_angle))
ax.set_xlabel('$r$')
ax.set_ylabel(r'$\theta$')
ax.set_zlabel(r'$\phi$')
plt.show()
