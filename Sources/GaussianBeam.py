import numpy as np
from .Beamlet import Beamlet
from .Source import Source
from Utils import *

class GaussianBeam(Source) :
    def __init__(self, power, radius, wavelength, num_rays, position, direction, polarization):
        super().__init__(power, radius, wavelength, num_rays, position, direction, polarization)
    
    def generate_rays(self):
        print("[Gaussian Source] :\tGenerating Rays...")

        N = self.number_of_rays
        u, v, w = self.local_frame

        r = np.sqrt(-self.radius**2 * np.log(np.random.rand(N)))
        phi = np.random.uniform(0, 2 * PI, N)

        x = r * np.cos(phi)
        y = r * np.sin(phi)

        origins = self.position[:, None] + (x * v[:, None]) + (y * u[:, None])
        origins = origins.T

        sigma = self.wavelength / (PI * self.radius)

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
                wavelength=self.wavelength
            )
            for i in range(N)
        ]

        print("[Gaussian Source] :\tDone.")

        return beam