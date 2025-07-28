import numpy as np
from Elements.OpticalElement import OpticalElement
from Utils import *

class Polarizer(OpticalElement):
    def __init__(self, position, orientation, name, diameter, transmission_angle):
        super().__init__(position, orientation, name)
        self.radius = diameter / 2
        self.theta = transmission_angle

        self.set_up_localframe()
    
    def set_up_localframe(self) :
        w = self.orientation

        tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

        u = np.cross(w, tmp)
        u /= np.linalg.norm(u)
        
        v = np.cross(w, u)

        # If w is too parallel to x-axis, swap u and v
        if abs(w[0]) > 0.9:
            u, v = v, u

        self.local_frame = (u, v, w)

    def hit(self, beamlet):
        if (np.abs(np.dot(beamlet.direction, self.orientation)) < 1e-6) : return False

        t = - np.dot((beamlet.position - self.position), self.orientation) / np.dot(beamlet.direction, self.orientation)
        if (t < 0) : return False

        if (np.linalg.norm(beamlet.position + t * beamlet.direction - self.position) >= self.radius) : return False

        return t
    
    def interact(self, beamlet):
        u, v, _ = self.local_frame

        transmission_axis = np.cos(self.theta) * v + np.sin(self.theta) * u

        if np.random.uniform(low = 0.0, high = 1.0) > np.abs(transmission_axis @ beamlet.polarization)**2:
            beamlet.active = False
        else:
            beamlet.polarization = transmission_axis
            beamlet.propagate(EPSILON)