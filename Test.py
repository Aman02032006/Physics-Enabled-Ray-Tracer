from Elements import *
import numpy as np

cos_theta_i = 1 / np.sqrt(2)
ri = 0.1568 + 1j * 3.806
ri = 1.5

sin_theta_i = np.lib.scimath.sqrt(1 - cos_theta_i**2)
sin_theta_t = (1 / ri) * sin_theta_i

theta_i = np.arcsin(sin_theta_i)
theta_t = np.arcsin(sin_theta_t)

rs = -np.sin(theta_i - theta_t) / np.sin(theta_i + theta_t)
rp = np.tan(theta_i - theta_t) / np.tan(theta_i + theta_t)

print(np.angle(rs), np.angle(rp))