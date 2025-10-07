import numpy as np
from Elements.OpticalElement import OpticalElement
from Utils import *

class Polarizer(OpticalElement):
    def __init__(self, position, orientation, name, diameter, transmission_angle):
        super().__init__(position, orientation, name)
        self.radius = diameter / 2
        self.theta = transmission_angle

        self.set_up_localframe()

        self.model_path = 'Models/Polarizer.ipt'
    
    def __iter__(self):
        yield self

    def set_up_localframe(self) :
        w = self.orientation

        tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

        u = normalize(cross(w, tmp))
        v = normalize(cross(w, u))

        # If w is too parallel to x-axis, swap u and v
        if abs(w[0]) > 0.9:
            u, v = v, u

        self.local_frame = (u, v, w)

    def hit(self, beamlet):
        if (np.abs(dot(beamlet.direction, self.orientation)) < 1e-6) : return False

        t = - dot(vec_sub(beamlet.position, self.position), self.orientation) / dot(beamlet.direction, self.orientation)
        if (t < 0) : return False

        if (norm(vec_sub(vec_add(beamlet.position, scale(beamlet.direction, t)), self.position)) >= self.radius) : return False

        return t
    
    def interact(self, beamlet):
        u, v, _ = self.local_frame

        transmission_axis = vec_add(scale(v, np.cos(self.theta)), scale(u, np.sin(self.theta)))

        if np.random.uniform(low = 0.0, high = 1.0) > np.abs(dot(transmission_axis, beamlet.polarization))**2:
            beamlet.active = False
        else:
            beamlet.polarization = transmission_axis
            beamlet.propagate(EPSILON)