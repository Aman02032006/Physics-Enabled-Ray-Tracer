import numpy as np
from Utils import *
from Elements.OpticalElement import OpticalElement

class HalfWavePlate(OpticalElement):
    def __init__(self, position, orientation, name, fast_axis_angle, diameter, retardance_error = 0.0, transmittivity = 1.0):
        super().__init__(position, orientation, name)
        self.radius = diameter / 2
        self.theta = fast_axis_angle
        self.retardance = PI * (1.0 + 2 * retardance_error)
        self.transmittivity = transmittivity

        self.set_up_localframe()
    
    def hit(self, beamlet):
        if (np.abs(np.dot(beamlet.direction, self.orientation)) < 1e-6) : return False

        t = - np.dot((beamlet.position - self.position), self.orientation) / np.dot(beamlet.direction, self.orientation)
        if (t < 0) : return False

        if (np.linalg.norm(beamlet.position + t * beamlet.direction - self.position) >= self.radius) : return False

        return t
    
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
    
    def axes(self):
        u, v, _ = self.local_frame
        fast_axis = np.cos(self.theta) * v + np.sin(self.theta) * u
        slow_axis = np.cos(self.theta) * u - np.sin(self.theta) * v

        return fast_axis, slow_axis

    
    def interact(self, beamlet):
        if np.random.rand() <= self.transmittivity:
            f, s = self.axes()

            E_in = beamlet.polarization
            Ef = np.dot(E_in, f)
            Es = np.dot(E_in, s)

            # print(f"[{self.name}] :\tInput Polarization = {beamlet.polarization}")
            beamlet.polarization = np.array(Ef * f + np.exp(I * self.retardance) * Es * s)
            # print(f"[{self.name}] :\tOutput Polarization = {beamlet.polarization}")

            beamlet.propagate(EPSILON)
        else:
            beamlet.active = False