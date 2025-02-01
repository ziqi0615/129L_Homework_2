#!/usr/bin/env python

from numpy import exp, log, sin, cos, arccos, pi, linspace, meshgrid, full_like, array, cross, outer
from matplotlib import pyplot as plt

def cartesian_coordinates(theta_angle, phi_angle):
	return (sin(theta_angle)*cos(phi_angle), sin(theta_angle)*sin(phi_angle), cos(theta_angle))

# a
def intersect_sphere(ax, f1, f2):
	t = linspace(-1, 1, 101)
	x1, y1, z1 = cartesian_coordinates(*f1(t))
	x2, y2, z2 = cartesian_coordinates(*f2(t))
	ax.plot(x1, y1, z1, color='b')
	ax.plot(x2, y2, z2, color='b')
	cx, cy, cz = x1[50], y1[50], z1[50]
	tan1 = array([x1[51]-cx, y1[51]-cy, z1[51]-cz])
	tan1 /= (tan1**2).sum()**0.5
	tan2 = array([x2[51]-cx, y2[51]-cy, z2[51]-cz])
	tan2 /= (tan2**2).sum()**0.5
	tx1, ty1, tz1 = tan1
	tx2, ty2, tz2 = tan2
	ax.plot([cx, cx+tx1/2], [cy, cy+ty1/2], [cz, cz+tz1/2], color='r')
	ax.plot([cx, cx+tx2/2], [cy, cy+ty2/2], [cz, cz+tz2/2], color='r')
	angle = arccos((tan1*tan2).sum())
	ax.text(cx, cy, cz, f'{angle:.2f}')

def intersect_stereo(ax, f1, f2):
	t = linspace(-1, 1, 101)
	x1, y1, z1 = cartesian_coordinates(*f1(t))
	x2, y2, z2 = cartesian_coordinates(*f2(t))
	x1, y1 = x1/(1-z1), y1/(1-z1)
	x2, y2 = x2/(1-z2), y2/(1-z2)
	ax.plot(x1, y1, color='b')
	ax.plot(x2, y2, color='b')
	cx, cy = x1[50], y1[50]
	tan1 = array([x1[51]-cx, y1[51]-cy])
	tan1 /= (tan1**2).sum()**0.5
	tan2 = array([x2[51]-cx, y2[51]-cy])
	tan2 /= (tan2**2).sum()**0.5
	tx1, ty1 = tan1
	tx2, ty2 = tan2
	ax.plot([cx, cx+tx1/2], [cy, cy+ty1/2], color='r')
	ax.plot([cx, cx+tx2/2], [cy, cy+ty2/2], color='r')
	angle = arccos((tan1*tan2).sum())
	ax.text(cx, cy, f'{angle:.2f}')

theta_angle = linspace(0, pi, 20)
phi_angle = linspace(0, 2*pi, 40)
theta_angle, phi_angle = meshgrid(theta_angle, phi_angle)
f1 = lambda t: (pi/2+pi/4*t, pi/16*t)
f2 = lambda t: (pi/2+pi/16*t, pi/4*t)
f3 = lambda t: (pi/4+pi/4*(exp(t)-1), pi+pi/4*t)
f4 = lambda t: (pi/4-pi/16*t, pi+pi/4*log(1+t/2))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(*cartesian_coordinates(theta_angle, phi_angle), color='grey')
intersect_sphere(ax, f1, f2)
intersect_sphere(ax, f3, f4)
plt.show()

fig, ax = plt.subplots()
x, y, z = cartesian_coordinates(theta_angle, phi_angle)
intersect_stereo(ax, f1, f2)
intersect_stereo(ax, f3, f4)
ax.set_xlabel("$x'$")
ax.set_ylabel("$y'$")
plt.show()
