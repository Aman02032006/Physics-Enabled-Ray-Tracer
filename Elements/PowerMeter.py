import matplotlib.pyplot as plt
import numpy as np
from Elements.OpticalElement import OpticalElement
from tqdm import tqdm
from Utils import *

class PowerMeter(OpticalElement) :
    def __init__(self, position, orientation, name, size, pixel_size = 0.0001):
        super().__init__(position, orientation, name)
        self.size = size
        self.pixel_size = pixel_size
        self.collected_beamlets = []
        self.collected_power = 0

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

        u, v, w = self.local_frame

        intersection_point = beamlet.position + t * beamlet.direction
        x_point = np.dot((intersection_point - self.position), v)
        y_point = np.dot((intersection_point - self.position), u)

        if abs(x_point) > self.size / 2 or abs(y_point) > self.size / 2 : return False

        return t
    
    def interact(self, beamlet):
        self.collected_beamlets.append(beamlet)
        self.collected_power += np.abs(beamlet.amplitude)**2
        beamlet.active = False
    
    def power(self):
        print(f"[{self.name}] :\tTotal Power on grid: {self.collected_power:.6f} W")

    def compute_wavefront(self):

        if not self.collected_beamlets:
            print(f"[{self.name}] :\tNo Beamlets collected.")
            return None, None, None, None, None

        half_extent = self.size / 2
        num_pixels = int(self.size / self.pixel_size)
        print(f"[{self.name}] :\tHalf Extent = {half_extent}, number of pixels = {num_pixels * num_pixels}")
        x = np.linspace(-half_extent, half_extent, num_pixels)
        y = np.linspace(-half_extent, half_extent, num_pixels)
        X, Y = np.meshgrid(x, y)

        # initializing complex field grids for Ex and Ey and Ez
        Ex_grid = np.zeros((num_pixels, num_pixels), dtype= complex)
        Ey_grid = np.zeros((num_pixels, num_pixels), dtype= complex)
        Ez_grid = np.zeros((num_pixels, num_pixels), dtype= complex)

        u, v, _ = self.local_frame
        sigma = 2 * self.pixel_size
        window = int(10 * sigma / self.pixel_size)

        for beamlet in tqdm(self.collected_beamlets, desc = "[Power Meter] :\tComputing Wavefront"):
            # Local detector-plane coordinates for beamlet hit
            dx = beamlet.position - self.position
            x0 = dx @ v
            y0 = dx @ u

            # Electric field of beam
            field = beamlet.E

            # Spatial Gaussian envelope centered at x0, y0
            # Get pixel indices
            i0 = int((x0 + half_extent) / self.pixel_size)
            j0 = int((y0 + half_extent) / self.pixel_size)

            # Define local patch bounds
            i_min = max(i0 - window, 0)
            i_max = min(i0 + window + 1, num_pixels)
            j_min = max(j0 - window, 0)
            j_max = min(j0 + window + 1, num_pixels)

            # Build local subgrid
            x_local = x[i_min:i_max]
            y_local = y[j_min:j_max]
            X_local, Y_local = np.meshgrid(x_local, y_local)
            envelope = np.exp(-((X_local - x0)**2 + (Y_local - y0)**2) / (2 * sigma**2))
            envelope /= (2 * np.pi * sigma**2)

            # Add contribution to patch of grid
            Ex_grid[j_min:j_max, i_min:i_max] += field[0] * envelope
            Ey_grid[j_min:j_max, i_min:i_max] += field[1] * envelope
            Ez_grid[j_min:j_max, i_min:i_max] += field[2] * envelope
        
        return X, Y, Ex_grid, Ey_grid, Ez_grid

    def total_intensity(self):
        _, _, Ex_grid, Ey_grid, Ez_grid = self.compute_wavefront()
        Power_Grid = (np.abs(Ex_grid)**2 + np.abs(Ey_grid)**2 + np.abs(Ez_grid)**2) * self.pixel_size**2
        return np.sum(Power_Grid)
    
    def plot(self):
        print(f"[{self.name}] :\t[1/4] Computing wavefront...")
        X, Y, Ex_grid, Ey_grid, Ez_grid = self.compute_wavefront()

        if Ex_grid is None or Ey_grid is None or Ez_grid is None:
            print(f"[{self.name}] :\tError: Field Intensity are not provided")
            return
        
        # Total intensity
        print(f"[{self.name}] :\t[2/4] Computing power grid...")
        Power_Grid = (np.abs(Ex_grid)**2 + np.abs(Ey_grid)**2 + np.abs(Ez_grid)**2) * self.pixel_size**2
        Power_Grid /= np.max(Power_Grid)

        self.power()

        print(f"[{self.name}] :\t[3/4] Plotting 2D power map...")
        plt.figure(figsize = (6, 6))
        extent = (X.min(), X.max(), Y.min(), Y.max())
        plt.imshow(Power_Grid, extent=extent, origin='lower', cmap='inferno', vmin = 0.0)
        plt.colorbar(label='Power')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.xlim((-self.size / 2, self.size / 2))
        plt.ylim((-self.size / 2, self.size / 2))
        plt.title(f"{self.name} Power Map")
        plt.gca().set_aspect('equal')
        plt.tight_layout()

        """
        print(f"[{self.name}] :\t[4/4] Plotting 1D cut...")
        # 1D Horizontal Cut
        mid_y_index = Power_Grid.shape[0] // 2
        x_coords = X[mid_y_index]
        power_cut = Power_Grid[mid_y_index]

        # --- Theoretical Curve Overlay ---
        w0 = 1e-3            # beam waist (adjust based on your source)
        wavelength = 633e-9  # meters
        z = np.linalg.norm(self.position)  # distance from source
        z_R = np.pi * w0**2 / wavelength
        wz = w0 * np.sqrt(1 + (z / z_R)**2)

        # Gaussian power profile: scale I0 to match peak of simulation
        I0 = np.max(power_cut)
        power_theory = I0 * np.exp(-2 * (x_coords**2) / wz**2)

        # Plot 1D cut with theory
        plt.figure(figsize=(8, 4))
        plt.plot(x_coords, power_cut, label='Simulated', color='blue')
        plt.plot(x_coords, power_theory, label='Theoretical Gaussian', linestyle='--', color='red')
        plt.xlabel("X (m)")
        plt.ylabel("Power per pixel (W)")
        plt.title(f"{self.name} Intensity 1D Cut")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        print("[✓] Done.")
        plt.show()"""