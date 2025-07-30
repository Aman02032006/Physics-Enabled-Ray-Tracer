import numpy as np
from Utils import *

class Source:
    def __init__(self, power, radius, wavelength, num_rays, position, direction, polarization):
        self.power = power
        self.radius = radius
        self.wavelength = wavelength
        self.number_of_rays = num_rays
        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.polarization = np.array(polarization) / np.linalg.norm(np.array(polarization))

        self.set_up_localframe()
    
    def set_up_localframe(self):
        w = self.direction

        tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(w, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(w, u)

        if abs(w[0]) > 0.9 :
            u, v = v, u
        
        self.local_frame = (u, v, w)

    def generate_rays(self):
        pass