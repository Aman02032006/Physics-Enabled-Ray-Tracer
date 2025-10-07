import numpy as np
from Elements.OpticalElement import OpticalElement
from Utils import *

class BeamSplitter(OpticalElement):
    def __init__(self, position, orientation, name, size, refractive_index, transmittivity = 0.5):
        super().__init__(position, orientation, name)
        self.size = size
        self.transmittivity = transmittivity
        self.refractive_index = refractive_index

        self.set_up_localframe()

        self.model_path = "Models\\Beam Splitter.iam"
    
    def __iter__(self):
        yield self

    def set_up_localframe(self) :
        w = self.orientation

        tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        
        u = normalize(cross(w, tmp))
        v = normalize(cross(w, u))
        
        if abs(w[0]) > 0.9:
            u, v = v, u

        self.local_frame = (u, v, w)
    
    def hit(self, beamlet):
        if (np.abs(dot(beamlet.direction, self.orientation)) < 1e-6) : return False

        t = - dot(vec_sub(beamlet.position, self.position), self.orientation) / dot(beamlet.direction, self.orientation)
        if (t < 0) : return False

        point_in_plane = vec_add(beamlet.position, scale(beamlet.direction, t))
        u, v, _ = self.local_frame
        if np.abs(dot(vec_sub(point_in_plane, self.position), u)) >= self.size / 2 or np.abs(dot(vec_sub(point_in_plane, self.position), v)) >= self.size / np.sqrt(2) :
            return False

        return t

    def interact(self, beamlet):
        if np.random.uniform(0, 1.0) < self.transmittivity:
            beamlet.amplitude *= 1 / np.sqrt(2)
            beamlet.propagate(EPSILON)
        else:
            n = self.orientation
            k_in = beamlet.direction

            s_hat = normalize(cross(n, k_in))
            p_hat = normalize(cross(s_hat, k_in))

            E_in = beamlet.polarization
            E_p = dot(E_in, p_hat)
            E_s = dot(E_in, s_hat)
            
            beamlet.direction = vec_sub(beamlet.direction, scale(self.orientation, 2 * dot(beamlet.direction, self.orientation)))

            k_out = beamlet.direction

            s_hat_out = normalize(cross(n, k_out))
            p_hat_out = normalize(cross(s_hat_out, k_out))

            beamlet.polarization = np.array(E_s * s_hat_out + E_p * p_hat_out)
            beamlet.amplitude *= 1 / np.sqrt(2)
            beamlet.elemental_phase += PI / 2

            beamlet.propagate(EPSILON)

