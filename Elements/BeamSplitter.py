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
    
    def set_up_localframe(self) :
        w = self.orientation

        tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        
        u = np.cross(w, tmp)
        u /= np.linalg.norm(u)
        
        v = np.cross(w, u)
        
        if abs(w[0]) > 0.9:
            u, v = v, u

        self.local_frame = (u, v, w)
    
    def hit(self, beamlet):
        if (np.abs(np.dot(beamlet.direction, self.orientation)) < 1e-6) : return False

        t = - np.dot((beamlet.position - self.position), self.orientation) / np.dot(beamlet.direction, self.orientation)
        if (t < 0) : return False

        point_in_plane = beamlet.position + t * beamlet.direction
        u, v, _ = self.local_frame
        if np.abs(np.dot((point_in_plane - self.position), u)) >= self.size / 2 or np.abs(np.dot((point_in_plane - self.position), v)) >= self.size / np.sqrt(2) :
            return False

        return t

    def interact(self, beamlet):
        if np.random.uniform(0, 1.0) < self.transmittivity:
            beamlet.amplitude *= 1 / np.sqrt(2)
            beamlet.propagate(EPSILON)
        else:
            n = self.orientation
            k_in = beamlet.direction

            s_hat = np.cross(n, k_in)
            s_hat /= np.linalg.norm(s_hat)
            p_hat = np.cross(s_hat, k_in)
            p_hat /= np.linalg.norm(p_hat)

            E_in = beamlet.polarization
            E_p = np.dot(E_in, p_hat)
            E_s = np.dot(E_in, s_hat)
            
            beamlet.direction = beamlet.direction - 2 * self.orientation * (beamlet.direction @ self.orientation)
            beamlet.beam_axis = beamlet.beam_axis - 2 * self.orientation * (beamlet.beam_axis @ self.orientation)

            k_out = beamlet.direction

            s_hat_out = np.cross(n, k_out)
            s_hat_out /= np.linalg.norm(s_hat_out)
            p_hat_out = np.cross(s_hat_out, k_out)
            p_hat_out /= np.linalg.norm(p_hat_out)

            beamlet.polarization = np.array(E_s * s_hat_out + E_p * p_hat_out)
            beamlet.amplitude *= 1 / np.sqrt(2)
            beamlet.elemental_phase += PI / 2

            beamlet.propagate(EPSILON)

