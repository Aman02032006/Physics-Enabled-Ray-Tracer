import numpy as np
from Elements.OpticalElement import OpticalElement
from Utils import *

class Mirror(OpticalElement) :
    def __init__(self, position, orientation, name, diameter, refractive_index = 0.1568 + I * 3.806, reflectivity = 1.0):
        super().__init__(position, orientation, name)
        self.radius = diameter / 2.0
        self.reflectivity = reflectivity
        self.refractive_index = refractive_index

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
    
    def fresnels_coefficients_reflection(self, ri, cos_theta_i):
        sin_theta_i = np.lib.scimath.sqrt(1 - cos_theta_i**2)
        sin_theta_t = (1 / ri) * sin_theta_i
        
        theta_i = np.arcsin(sin_theta_i)
        theta_t = np.arcsin(sin_theta_t)

        rs = -np.sin(theta_i - theta_t) / np.sin(theta_i + theta_t)
        rp = np.tan(theta_i - theta_t) / np.tan(theta_i + theta_t)

        return rs, rp

    def interact(self, beamlet):
        if np.random.rand() > self.reflectivity : 
            beamlet.active = False
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

            cos_theta_i = - k_in @ n

            rs, rp = self.fresnels_coefficients_reflection(self.refractive_index, cos_theta_i)

            beamlet.direction = beamlet.direction - 2 * self.orientation * (beamlet.direction @ self.orientation)
            beamlet.beam_axis = beamlet.beam_axis - 2 * self.orientation * (beamlet.beam_axis @ self.orientation)

            k_out = beamlet.direction

            s_hat_out = np.cross(n, k_out)
            s_hat_out /= np.linalg.norm(s_hat_out)
            p_hat_out = np.cross(s_hat_out, k_out)
            p_hat_out /= np.linalg.norm(p_hat_out)

            beamlet.polarization = np.array(rs * E_s * s_hat_out + rp * E_p * p_hat_out)

            beamlet.propagate(EPSILON)
