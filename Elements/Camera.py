import matplotlib.pyplot as plt
import numpy as np
from Elements.OpticalElement import OpticalElement
from tqdm import tqdm
from Utils import *
from scipy.special import genlaguerre

class Camera(OpticalElement) :
    def __init__(self, position, orientation, name, size, pixel_size = 0.0001):
        super().__init__(position, orientation, name)
        self.size = size
        self.pixel_size = pixel_size
        self.pixel_count = int(size / pixel_size)
        self.wavefront = np.zeros((self.pixel_count, self.pixel_count, 3), dtype = complex)

        self.set_up_localframe()

        self.model_path = 'Models/Power Meter.ipt'
    
    def __iter__(self):
        yield self
    
    def clear(self):
        self.wavefront = np.zeros((self.pixel_count, self.pixel_count, 3), dtype = complex)

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
        u, v, w = self.local_frame
        x = dot(vec_sub(beamlet.position, self.position), v)
        y = dot(vec_sub(beamlet.position, self.position), u)

        x += self.size / 2
        y += self.size / 2

        i, j = int(x / self.pixel_size), int(y / self.pixel_size)

        self.wavefront[j, i, :] += beamlet.E
        beamlet.active = False
    
    def plot(self, p = 0, l = 0):
        x = np.linspace(-self.size / 2, self.size / 2, self.pixel_count)
        y = np.linspace(-self.size / 2, self.size / 2, self.pixel_count)
        X, Y = np.meshgrid(x, y)

        Intensity_Map = np.abs(self.wavefront[:, :, 0])**2 + np.abs(self.wavefront[:, :, 1])**2 + np.abs(self.wavefront[:, :, 2])**2

        plt.figure(figsize = (6, 6))
        extent = (X.min(), X.max(), Y.min(), Y.max())
        plt.imshow(Intensity_Map, extent=extent, origin='lower', cmap='inferno', vmin = 0.0)
        plt.colorbar(label='Power')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.xlim((-self.size / 2, self.size / 2))
        plt.ylim((-self.size / 2, self.size / 2))
        plt.title(f"{self.name} Intensity Map")
        plt.gca().set_aspect('equal')
        plt.tight_layout()