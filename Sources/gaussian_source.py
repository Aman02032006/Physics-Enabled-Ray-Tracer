import numpy as np
import matplotlib.pyplot as plt
from Sources.beamlet import Beamlet
from Utils import *
from tqdm import tqdm

class GaussianSource :
    def __init__(self, power, waist_radius, wavelength, num_rays, center, direction, polarization) :
        self.power = power
        self.waist_radius = waist_radius
        self.wavelength = wavelength
        self.center = np.array(center)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.polarization = np.array(polarization) / np.linalg.norm(np.array(polarization))
        self.number_of_rays = num_rays

        self.theta_div = self.wavelength / (PI * self.waist_radius)

        self.set_up_localframe()
    
    def set_up_localframe(self) :
        w = self.direction

        tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(w, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(w, u)
        self.local_frame = (u, v, w)
    
    def generate_rays(self):
        print("[Gaussian Source] :\tGenerating Rays...")

        N = self.number_of_rays
        u, v, w = self.local_frame

        r = np.sqrt(-self.waist_radius**2 * np.log(np.random.rand(N)))
        phi = np.random.uniform(0, 2 * PI, N)

        x = r * np.cos(phi)
        y = r * np.sin(phi)

        origins = self.center[:, None] + (x * v[:, None]) + (y * u[:, None])
        origins = origins.T

        sigma = self.theta_div  

        theta_x = np.random.normal(loc=0.0, scale=sigma, size=N)
        theta_y = np.random.normal(loc=0.0, scale=sigma, size=N)

        dz = np.sqrt(1 - theta_x**2 - theta_y**2)

        directions = (theta_x[:,None] * v + theta_y[:,None] * u + dz[:,None] * w)
        directions /= np.linalg.norm(directions, axis=1)[:,None]

        amplitude = np.sqrt(self.power / N)
        polarization = self.polarization

        beam = [
            Beamlet(
                position=origins[i],
                direction=directions[i],
                amplitude=amplitude,
                phase = 0.0,
                polarization=polarization[0] * v + polarization[1] * u + polarization[2] * w,
                wavelength=self.wavelength,
                beam_axis=self.direction
            )
            for i in range(N)
        ]

        print("[Gaussian Source] :\tDone.")

        return beam
