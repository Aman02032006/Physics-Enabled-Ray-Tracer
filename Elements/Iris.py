import numpy as np
import matplotlib.pyplot as plt
from Elements.OpticalElement import OpticalElement
from Sources import *
from Utils import *
from tqdm import tqdm

class Iris(OpticalElement):
    def __init__(self, position, orientation, name, radius, size):
        super().__init__(position, orientation, name)
        self.radius = radius
        self.size = size
        self.collected_beamlets = []

        self.set_up_localframe()
    
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

        u, v, _ = self.local_frame

        intersection_point = vec_add(beamlet.position, scale(beamlet.direction, t))
        x_point = dot(vec_sub(intersection_point, self.position), v)
        y_point = dot(vec_sub(intersection_point, self.position), u)

        if abs(x_point) > self.size / 2 or abs(y_point) > self.size / 2 : return False

        return t
    
    def interact(self, beamlet):
        self.collected_beamlets.append(beamlet)
        beamlet.active = False

    def diffract(self, delta = 0.000025, z = 0.005):

        u, v, w = self.local_frame
        
        num_pixels = int(self.size / (2 * delta))
        x = np.linspace(-self.size / 2, self.size / 2, num_pixels)
        y = np.linspace(-self.size / 2, self.size / 2, num_pixels)
        X, Y = np.meshgrid(x, y)

        Ex_grid = np.zeros((num_pixels, num_pixels), dtype= complex)
        Ey_grid = np.zeros((num_pixels, num_pixels), dtype= complex)

        sigma = 2 * delta

        for beamlet in tqdm(self.collected_beamlets, desc = "[Iris] : Constructing wavefront"):
            pos = vec_sub(beamlet.position, self.position)
            x0 = dot(pos, v)
            y0 = dot(pos, u)

            field = beamlet.E

            envelope = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

            Ex_grid += dot(field, v) * envelope
            Ey_grid += dot(field, u) * envelope

        R = np.sqrt(X**2 + Y**2)
        mask = (R <= self.radius)
        # mask = (np.abs(X) <= 0.0001) & (np.abs(Y) <= 0.001)

        Ex_grid *= mask
        Ey_grid *= mask

        self.plot_intensity_map(X = X, Y = Y, Ex = Ex_grid, Ey = Ey_grid, title = "Before Diffraction")
        
        Ex_grid = propagate_angular_spectrum(Ex_grid, delta, delta, beamlet.k_mag, z)
        Ey_grid = propagate_angular_spectrum(Ey_grid, delta, delta, beamlet.k_mag, z)

        self.plot_intensity_map(X = X, Y = Y, Ex = Ex_grid, Ey = Ey_grid, title = "After Diffraction")
        
        I = (np.abs(Ex_grid))**2 + (np.abs(Ey_grid))**2
        P = I / I.sum()
        P_flat = P.ravel()
        P_flat /= P_flat.sum()
        dphi_dx, dphi_dy = field_phase_gradients(Ex_grid, delta, delta)

        diffracted_beam = []

        for beamlet in tqdm(self.collected_beamlets, desc = "[Iris] : Modelling Rays"):
            r = norm(vec_sub(beamlet.position, self.position))
            if r > self.radius : 
                beamlet.active = False
                continue

            chosen_idx = np.random.choice(P_flat.size, p=P_flat)
            iy, ix = np.unravel_index(chosen_idx, P.shape)
            x_p, y_p = x[ix], y[iy]

            beamlet.direction = normalize(vec_sub(np.array([x_p, y_p, z]), beamlet.position))
        
        return self.collected_beamlets

    def plot_intensity_map(self, X, Y, Ex, Ey, title):
        Power_Grid = (np.abs(Ex)**2 + np.abs(Ey)**2)
        # Power_Grid /= np.max(Power_Grid)

        plt.figure(figsize = (6, 6))
        extent = (X.min(), X.max(), Y.min(), Y.max())
        plt.imshow(Power_Grid, extent=extent, origin='lower', cmap='cividis', vmin = 0.0)
        plt.colorbar(label='Power')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.xlim((-self.size / 2, self.size / 2))
        plt.ylim((-self.size / 2, self.size / 2))
        plt.title(f"{self.name} {title}")
        plt.gca().set_aspect('equal')
        plt.tight_layout()

    def clear(self):
        self.collected_beamlets = []