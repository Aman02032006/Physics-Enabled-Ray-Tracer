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
        a = dot(beamlet.direction, beamlet.direction)
        b = 2 * dot(vec_sub(beamlet.position, self.position), beamlet.direction)
        c = dot(vec_sub(beamlet.position, self.position), vec_sub(beamlet.position, self.position)) - self.radius**2

        D = b**2 - 4 * a * c
        if D <= 0.0:
            return False

        D = np.sqrt(D)

        if self.radius < 0:
            t = (-b + D) / (2 * a)
        else:
            t = (-b - D) / (2 * a)

        if t < 0:
            return False

        intersection_point = vec_add(beamlet.position, scale(beamlet.direction, t))
        r = norm(cross(vec_sub(intersection_point, self.position), self.orientation))

        if r > self.aperture / 2:
            return False

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
        normal = normalize(vec_sub(beamlet.position, self.position))

        if dot(normal, beamlet.direction) < 0.0:
            ni, nt = self.n_out, self.n_in
        else:
            ni, nt = self.n_in, self.n_out
            normal = scale(normal, -1)

        cos_theta_i = -dot(normal, beamlet.direction)
        n = ni / nt

        if n * np.sqrt(1 - cos_theta_i**2) > 1:
            # Reflection
            k_in = beamlet.direction
            s_hat = normalize(cross(normal, k_in))
            p_hat = normalize(cross(s_hat, k_in))

            E_in = beamlet.polarization
            E_p = dot(E_in, p_hat)
            E_s = dot(E_in, s_hat)

            rs, rp = self.fresnels_coefficients_reflection(n, cos_theta_i)

            beamlet.direction = vec_sub(
                beamlet.direction,
                scale(normal, 2 * dot(beamlet.direction, normal))
            )

            k_out = beamlet.direction
            s_hat_out = normalize(cross(normal, k_out))
            p_hat_out = normalize(cross(s_hat_out, k_out))

            beamlet.polarization = vec_add(scale(s_hat_out, rs * E_s), scale(p_hat_out, rp * E_p))

            beamlet.propagate(EPSILON)

        else:
            # Transmission
            k_in = beamlet.direction
            s_hat = normalize(cross(normal, k_in))
            p_hat = normalize(cross(s_hat, k_in))

            E_in = beamlet.polarization
            E_p = dot(E_in, p_hat)
            E_s = dot(E_in, s_hat)

            cos_theta_t = np.sqrt(1 - (1 - cos_theta_i**2) * n**2)

            beamlet.direction = normalize(vec_add(scale(beamlet.direction, n), scale(normal, n * cos_theta_i - cos_theta_t)))

            k_out = beamlet.direction
            s_hat_out = normalize(cross(normal, k_out))
            p_hat_out = normalize(cross(s_hat_out, k_out))

            beamlet.polarization = vec_add(scale(s_hat_out, E_s), scale(p_hat_out, E_p))
            beamlet.k_mag /= n

            beamlet.propagate(EPSILON)
        
class PlaneSurface(OpticalElement):
    def __init__(self, position, orientation, name, aperture, n_in, n_out):
        super().__init__(position, orientation, name)
        self.aperture = aperture
        self.n_in = n_in
        self.n_out = n_out
    
    def hit(self, beamlet):
        if abs(dot(beamlet.direction, self.orientation)) < 1e-6:
            return False

        t = -dot(vec_sub(beamlet.position, self.position), self.orientation) / dot(beamlet.direction, self.orientation)
        if t < 0:
            return False

        intersection_point = vec_add(beamlet.position, scale(beamlet.direction, t))
        if norm(vec_sub(intersection_point, self.position)) >= self.aperture / 2:
            return False

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
        cos_theta_i = dot(beamlet.direction, self.orientation)

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

            s_hat = normalize(cross(normal, k_in))
            p_hat = normalize(cross(s_hat, k_in))

            E_in = beamlet.polarization
            E_p = dot(E_in, p_hat)
            E_s = dot(E_in, s_hat)

            rs, rp = self.fresnels_coefficients_reflection(n, cos_theta_i)

            # print(f"[{self.name}] :\tBeamlet direction before reflection = {beamlet.direction}")
            beamlet.direction = vec_sub(beamlet.direction, scale(normal, 2 *dot(beamlet.direction, normal)))
            # print(f"[{self.name}] :\tBeamlet direction after reflection = {beamlet.direction}")

            k_out = beamlet.direction

            s_hat_out = normalize(cross(normal, k_out))
            p_hat_out = normalize(cross(s_hat_out, k_out))

            beamlet.polarization = np.array(rs * E_s * s_hat_out + rp * E_p * p_hat_out)

            beamlet.propagate(EPSILON)
        else:
            # print(f"[{self.name}] :\tBeamlet Transmitted")
            k_in = beamlet.direction

            s_hat = normalize(cross(normal, k_in))
            p_hat = normalize(cross(s_hat, k_in))

            E_in = beamlet.polarization
            E_p = dot(E_in, p_hat)
            E_s = dot(E_in, s_hat)

            cos_theta_t = np.sqrt(1 - (1 - cos_theta_i**2) * n**2)

            # print(f"[{self.name}] :\tBeamlet direction before transmission = {beamlet.direction}")
            beamlet.direction = normalize(vec_add(scale(beamlet.direction, n), scale(normal, (n * cos_theta_i - cos_theta_t))))
            # print(f"[{self.name}] :\tBeamlet direction after transmission = {beamlet.direction}")

            k_out = beamlet.direction

            s_hat_out = normalize(cross(normal, k_out))
            p_hat_out = normalize(cross(s_hat_out, k_out))

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

        self.model_path = 'Models/Bi Convex Lens.ipt'

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

        self.model_path = 'Models/Plano Convex Lens.ipt'

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