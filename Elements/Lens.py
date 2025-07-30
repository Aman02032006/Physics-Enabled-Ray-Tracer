import numpy as np
from Utils import *
from Elements.OpticalElement import OpticalElement
from Tracer import Tracer

class SphericalSurface(OpticalElement):
    def __init__(self, center, orientation, name, radius, aperture, n_in, n_out):
        super().__init__(center, orientation, name)
        self.radius = radius
        self.aperture = aperture
        self.n_in = n_in
        self.n_out = n_out
    
    def hit(self, beamlet):
        # print(f"[{self.name}] :\tHit checked")
        a = np.dot(beamlet.direction, beamlet.direction)
        b = 2 * np.dot(beamlet.position - self.position, beamlet.direction)
        c = np.dot(beamlet.position - self.position, beamlet.position - self.position) - self.radius**2

        D = b**2 - 4 * a * c

        if D <= 0.0 : return False

        D = np.sqrt(D)

        if self.radius < 0 :
            t = (-b + D) / (2 * a)
        else :
            t = (-b - D) / (2 * a)

        if t < 0 : return False

        intersection_point = beamlet.position + t * beamlet.direction

        r = np.linalg.norm(np.cross(intersection_point - self.position, self.orientation))

        if r > self.aperture / 2 : return False

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
        normal = beamlet.position - self.position
        normal /= np.linalg.norm(normal)

        if np.dot(normal, beamlet.direction) < 0.0 :
            ni, nt = self.n_out, self.n_in
        else :
            ni, nt = self.n_in, self.n_out
            normal = -normal

        # print(f"[{self.name}] :\tni = {ni}, nt = {nt}")
        
        cos_theta_i = -np.dot(normal, beamlet.direction)

        n = ni / nt

        if n * np.sqrt(1 - cos_theta_i**2) > 1:
            # print(f"[{self.name}] :\tBeamlet Reflected")
            k_in = beamlet.direction

            s_hat = np.cross(normal, k_in)
            s_hat /= np.linalg.norm(s_hat)
            p_hat = np.cross(s_hat, k_in)
            p_hat /= np.linalg.norm(p_hat)

            E_in = beamlet.polarization
            E_p = np.dot(E_in, p_hat)
            E_s = np.dot(E_in, s_hat)

            rs, rp = self.fresnels_coefficients_reflection(n, cos_theta_i)

            # print(f"[{self.name}] :\tBeamlet direction before reflection = {beamlet.direction}")
            beamlet.direction = beamlet.direction - 2 * normal * (beamlet.direction @ normal)
            # print(f"[{self.name}] :\tBeamlet direction after reflection = {beamlet.direction}")

            k_out = beamlet.direction

            s_hat_out = np.cross(normal, k_out)
            s_hat_out /= np.linalg.norm(s_hat_out)
            p_hat_out = np.cross(s_hat_out, k_out)
            p_hat_out /= np.linalg.norm(p_hat_out)

            beamlet.polarization = np.array(rs * E_s * s_hat_out + rp * E_p * p_hat_out)

            beamlet.propagate(EPSILON)
        else:
            # print(f"[{self.name}] :\tBeamlet Transmitted")
            k_in = beamlet.direction

            s_hat = np.cross(normal, k_in)
            s_hat /= np.linalg.norm(s_hat)
            p_hat = np.cross(s_hat, k_in)
            p_hat /= np.linalg.norm(p_hat)

            E_in = beamlet.polarization
            E_p = np.dot(E_in, p_hat)
            E_s = np.dot(E_in, s_hat)

            cos_theta_t = np.sqrt(1 - (1 - cos_theta_i**2) * n**2)

            ts = 2 * ni * cos_theta_i / (ni * cos_theta_i + nt * cos_theta_t)
            tp = 2 * ni * cos_theta_i / (nt * cos_theta_i + ni * cos_theta_t)

            # print(f"[{self.name}] :\tBeamlet direction before transmission = {beamlet.direction}")
            beamlet.direction = n * beamlet.direction + (n * cos_theta_i - cos_theta_t) * normal
            beamlet.direction /= np.linalg.norm(beamlet.direction)
            # print(f"[{self.name}] :\tBeamlet direction after transmission = {beamlet.direction}")

            k_out = beamlet.direction

            s_hat_out = np.cross(normal, k_out)
            s_hat_out /= np.linalg.norm(s_hat_out)
            p_hat_out = np.cross(s_hat_out, k_out)
            p_hat_out /= np.linalg.norm(p_hat_out)

            beamlet.polarization = np.array(E_s * s_hat_out + E_p * p_hat_out)
            beamlet.k_mag /= n
            # print(f"[{self.name}] :\tn = {n}")

            beamlet.propagate(EPSILON)
        
class PlaneSurface(OpticalElement):
    def __init__(self, position, orientation, name, aperture, n_in, n_out):
        super().__init__(position, orientation, name)
        self.aperture = aperture
        self.n_in = n_in
        self.n_out = n_out
    
    def hit(self, beamlet):
        if (np.abs(np.dot(beamlet.direction, self.orientation)) < 1e-6) : return False

        t = - np.dot((beamlet.position - self.position), self.orientation) / np.dot(beamlet.direction, self.orientation)
        if (t < 0) : return False

        if (np.linalg.norm(beamlet.position + t * beamlet.direction - self.position) >= self.aperture / 2) : return False

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
        cos_theta_i = np.dot(beamlet.direction, self.orientation)

        if cos_theta_i > 0 :
            ni, nt = self.n_in, self.n_out
            normal = -self.orientation
        else :
            ni, nt = self.n_out, self.n_in
            normal = self.orientation
            cos_theta_i = -cos_theta_i
        
        # print(f"[{self.name}] :\tni = {ni}, nt = {nt}")

        n = ni / nt

        if n * np.sqrt(1 - cos_theta_i**2) > 1:
            # print(f"[{self.name}] :\tBeamlet Reflected")
            k_in = beamlet.direction

            s_hat = np.cross(normal, k_in)
            s_hat /= np.linalg.norm(s_hat)
            p_hat = np.cross(s_hat, k_in)
            p_hat /= np.linalg.norm(p_hat)

            E_in = beamlet.polarization
            E_p = np.dot(E_in, p_hat)
            E_s = np.dot(E_in, s_hat)

            rs, rp = self.fresnels_coefficients_reflection(n, cos_theta_i)

            # print(f"[{self.name}] :\tBeamlet direction before reflection = {beamlet.direction}")
            beamlet.direction = beamlet.direction - 2 * normal * np.dot(beamlet.direction, normal)
            # print(f"[{self.name}] :\tBeamlet direction after reflection = {beamlet.direction}")

            k_out = beamlet.direction

            s_hat_out = np.cross(normal, k_out)
            s_hat_out /= np.linalg.norm(s_hat_out)
            p_hat_out = np.cross(s_hat_out, k_out)
            p_hat_out /= np.linalg.norm(p_hat_out)

            beamlet.polarization = np.array(rs * E_s * s_hat_out + rp * E_p * p_hat_out)

            beamlet.propagate(EPSILON)
        else:
            # print(f"[{self.name}] :\tBeamlet Transmitted")
            k_in = beamlet.direction

            s_hat = np.cross(normal, k_in)
            s_hat /= np.linalg.norm(s_hat)
            p_hat = np.cross(s_hat, k_in)
            p_hat /= np.linalg.norm(p_hat)

            E_in = beamlet.polarization
            E_p = np.dot(E_in, p_hat)
            E_s = np.dot(E_in, s_hat)

            cos_theta_t = np.sqrt(1 - (1 - cos_theta_i**2) * n**2)

            ts = 2 * ni * cos_theta_i / (ni * cos_theta_i + nt * cos_theta_t)
            tp = 2 * ni * cos_theta_i / (nt * cos_theta_i + ni * cos_theta_t)

            # print(f"[{self.name}] :\tBeamlet direction before transmission = {beamlet.direction}")
            beamlet.direction = n * beamlet.direction + (n * cos_theta_i - cos_theta_t) * normal
            beamlet.direction /= np.linalg.norm(beamlet.direction)
            # print(f"[{self.name}] :\tBeamlet direction after transmission = {beamlet.direction}")

            k_out = beamlet.direction

            s_hat_out = np.cross(normal, k_out)
            s_hat_out /= np.linalg.norm(s_hat_out)
            p_hat_out = np.cross(s_hat_out, k_out)
            p_hat_out /= np.linalg.norm(p_hat_out)

            beamlet.polarization = np.array(E_s * s_hat_out + E_p * p_hat_out)
            beamlet.k_mag /= n
            # print(f"[{self.name}] :\tn = {n}")

            beamlet.propagate(EPSILON)    

class BiConvexLens(OpticalElement):
    def __init__(self, position, orientation, name, refractive_index, f_value, aperture):
        super().__init__(position, orientation, name)
        self.n = refractive_index
        self.f = f_value
        self.aperture = aperture

        r = 2 * self.f * (self.n - 1)
        t = r - np.sqrt(r**2 - 0.25 * self.aperture**2)

        print(f"[{self.name}] :\tr = {r}, t = {t}")

        self.Surface1 = SphericalSurface(center = self.position + (r - t - 0.002) * self.orientation, orientation = self.orientation, name = "Front Surface " + self.name, radius = r, aperture = self.aperture, n_in = self.n, n_out = 1.0)
        self.Surface2 = SphericalSurface(center = self.position - (r - t) * self.orientation, orientation = self.orientation, name = "Back Surface " + self.name, radius = -r, aperture = self.aperture, n_in = self.n, n_out = 1.0)

    def __iter__(self):
        yield self.Surface1
        yield self.Surface2

class PlanoConvexLens(OpticalElement):
    def __init__(self, position, orientation, name, refractive_index, f_value, aperture, flipped = False):
        super().__init__(position, orientation, name)
        self.n = refractive_index
        self.f = f_value
        self.aperture = aperture

        r = self.f * (self.n - 1)
        t = r - np.sqrt(r**2 - 0.25 * self.aperture**2)

        print(f"[{self.name}] :\tr = {r}, t = {t}")

        if not flipped :
            self.Surface1 = SphericalSurface(center = self.position + (r - t) * self.orientation, orientation = self.orientation, name = "Front Curved Surface " + self.name, radius = r, aperture = self.aperture, n_in = self.n, n_out = 1.0)
            self.Surface2 = PlaneSurface(position = self.position + 0.002 * self.orientation, orientation = self.orientation, name = "Back Planar Surface " + self.name, aperture = self.aperture, n_in = self.n, n_out = 1.0)
        else :
            self.Surface1 = PlaneSurface(position = self.position - 0.002 * self.orientation, orientation = self.orientation, name = "Front Planar Surface " + self.name, aperture = self.aperture, n_in = 1.0, n_out = self.n)
            self.Surface2 = SphericalSurface(center = self.position - (r - t) * self.orientation, orientation = self.orientation, name = "Back Curved Surface " + self.name, radius = -r, aperture = self.aperture, n_in = self.n, n_out = 1.0)

    def __iter__(self):
        yield self.Surface1
        yield self.Surface2