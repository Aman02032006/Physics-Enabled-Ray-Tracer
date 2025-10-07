import numpy as np
from Utils import *
from Elements.OpticalElement import OpticalElement

class QuarterWavePlate(OpticalElement):
    def __init__(self, position, orientation, fast_axis_angle, diameter, name, retardance_error = 0.0, transmittivity = 1.0):
        super().__init__(position, orientation, name)
        self.radius = diameter / 2
        self.theta = fast_axis_angle
        self.retardance = PI * (0.5 + 2 * retardance_error)
        self.transmittivity = transmittivity

        self.set_up_localframe()

        self.model_path = 'Models/Quarter Wave Plate.ipt'
    
    def __iter__(self):
        yield self

    def hit(self, beamlet):
        if (np.abs(dot(beamlet.direction, self.orientation)) < 1e-6) : return False

        t = - dot(vec_sub(beamlet.position, self.position), self.orientation) / dot(beamlet.direction, self.orientation)
        if (t < 0) : return False

        if (norm(vec_sub(vec_add(beamlet.position, scale(beamlet.direction, t)), self.position)) >= self.radius) : return False

        return t
    
    def set_up_localframe(self) :
        w = self.orientation

        tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

        u = normalize(cross(w, tmp))
        v = normalize(cross(w, u))

        # If w is too parallel to x-axis, swap u and v
        if abs(w[0]) > 0.9:
            u, v = v, u

        self.local_frame = (u, v, w)
    
    def axes(self):
        u, v, _ = self.local_frame
        fast_axis = vec_add(scale(v, np.cos(self.theta)), scale(u, np.sin(self.theta)))
        slow_axis = vec_sub(scale(u, np.cos(self.theta)), scale(v, np.sin(self.theta)))

        return fast_axis, slow_axis

    
    def interact(self, beamlet):
        if np.random.rand() <= self.transmittivity:
            f, s = self.axes()

            E_in = beamlet.polarization
            Ef = dot(E_in, f)
            Es = dot(E_in, s)

            # print(f"[{self.name}] :\t Input polarization = {beamlet.polarization}")
            beamlet.polarization = np.array(Ef * f + np.exp(I * self.retardance) * Es * s)
            # print(f"[{self.name}] :\t Output polarization = {beamlet.polarization}")

            beamlet.propagate(EPSILON)
        else:
            beamlet.active = False