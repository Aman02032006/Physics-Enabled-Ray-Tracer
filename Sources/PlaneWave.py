import numpy as np
from Utils import *
from .Source import Source
from .Beamlet import Beamlet

class PlaneWave(Source):
    def __init__(self, power, radius, wavelength, num_rays, position, direction, polarization):
        super().__init__(power, radius, wavelength, num_rays, position, direction, polarization)

        self.model_path = "Models\\Phase Meter.ipt"
    
    def generate_rays(self):
        print("[Plane Wave] :\tGenerating Rays...")

        u, v, w = self.local_frame

        # Grid size set so total number of inner points approximates your ray count
        grid_size = int(np.ceil(np.sqrt(self.number_of_rays * 4 / PI)))
        x = np.linspace(-self.radius, self.radius, grid_size)
        y = np.linspace(-self.radius, self.radius, grid_size)
        x_grid, y_grid = np.meshgrid(x, y)

        # Flatten for filtering
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()

        # Filter points inside the aperture
        inside_circle = x_flat**2 + y_flat**2 <= self.radius**2
        x_uniform = x_flat[inside_circle]
        y_uniform = y_flat[inside_circle]

        # If there are more points than rays needed, truncate
        if len(x_uniform) > self.number_of_rays:
            x_uniform = x_uniform[:self.number_of_rays]
            y_uniform = y_uniform[:self.number_of_rays]

        # Compute origins on this uniform grid
        origins = self.position + np.outer(x_uniform, v) + np.outer(y_uniform, u)
        num_points = len(origins)

        amplitude = np.sqrt(self.power / self.number_of_rays)
        polarization = self.polarization[0] * v + self.polarization[1] * u + self.polarization[2] * w

        beam = [
            Beamlet(
                position = origins[i],
                direction = self.direction,
                amplitude = amplitude,
                phase = 0.0,
                polarization = polarization,
                wavelength = self.wavelength
            )
            for i in range(num_points)
        ]

        print("[Plane Wave] :\tDone.")

        return beam
