import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from .Beamlet import Beamlet
from .Source import Source
from Utils import *

class LaguerreGaussianBeam(Source) :
    def __init__(self, power, radius, wavelength, num_rays, position, direction, polarization, l, p):
        super().__init__(power, radius, wavelength, num_rays, position, direction, polarization)
        self.l = l
        self.p = p
        self.L = genlaguerre(p, np.abs(self.l))

    def r_pdf(self, r):
        x = 2 * (r ** 2) / (self.radius ** 2)
        return (r * x ** np.abs(self.l) / self.radius) * (self.L(x) ** 2) * np.exp(-x)
    
    def k_pdf(self, k):
        rho = (self.radius**2 * k**2) / 2.0
        return (k * rho ** np.abs(self.l)) * (self.L(rho)**2) * np.exp(-rho)

    def plot_sampled(self, r, r_max):
        # Plot histogram of sampled r values
        plt.figure(figsize=(6,4))
        plt.hist(r, bins=100, density=True, alpha=0.6, label="Sampled distribution")

        # Overlay theoretical PDF (normalized)
        r_plot = np.linspace(0, r_max * self.radius, 500)
        pdf_vals = self.r_pdf(r_plot)
        pdf_vals /= np.trapezoid(pdf_vals, r_plot)   # normalize to match histogram
        plt.plot(r_plot, pdf_vals, 'r-', lw=2, label="Expected PDF")

        plt.xlabel("r")
        plt.ylabel("Frequency (normalized)")
        plt.title("Radial sampling distribution")
        plt.legend()
        plt.show()

    def generate_rays(self):
        print("[Laguerre Gaussian Source] :\tGenerating rays...")
        u, v, w = self.local_frame

        r_max = 5.0

        r_vals = np.linspace(0, r_max * self.radius, 1000)
        pdf_vals = self.r_pdf(r_vals)

        pdf_vals /= np.trapezoid(pdf_vals, r_vals)

        cdf_vals = cumulative_trapezoid(pdf_vals, r_vals, initial = 0)
        cdf_vals /= cdf_vals[-1]

        inverse_cdf = interp1d(cdf_vals, r_vals, bounds_error=False, fill_value=(r_vals[0], r_vals[-1]))

        u_vals = np.random.rand(self.number_of_rays)
        r = inverse_cdf(u_vals)

        self.plot_sampled(r=r, r_max=r_max)
        
        phi = np.random.uniform(0, 2 * PI, self.number_of_rays) 

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        origins = np.stack([x, y, np.zeros_like(x)], axis = 1) + self.position

        k = 2 * PI / self.wavelength
        k_perp = []
        k_perp_max = 5.0 / self.radius
        kvals = np.linspace(0, k_perp_max, 1000)
        p_max = np.max(self.k_pdf(kvals))

        while len(k_perp) < self.number_of_rays:
            k_try = np.random.uniform(0, k_perp_max, size = 10 * self.number_of_rays)
            p_try = np.random.uniform(0, p_max, size = 10 * self.number_of_rays)

            accepted = k_try[p_try < self.k_pdf(k_try)]

            k_perp.extend(accepted.tolist())
        
        k_perp = np.array(k_perp[:self.number_of_rays])
        phi_k = np.random.uniform(0, 2 * PI, self.number_of_rays)

        kx = k_perp * np.cos(phi_k)
        ky = k_perp * np.sin(phi_k)
        kz = np.sqrt(np.maximum(0, k**2 - kx**2 - ky**2))

        directions = (kx[:,None] * v + ky[:,None] * u + kz[:,None] * w)
        directions /= np.linalg.norm(directions, axis=1)[:,None]

        x = 2 * (r ** 2) / (self.radius ** 2)
        radial_sign = np.sign(self.L(x))
        phases = self.l * (phi)  + (radial_sign < 0) * PI

        amplitudes = np.sqrt(self.power / self.number_of_rays)
        polarization = self.polarization

        beam = [
            Beamlet(
                position = origins[i],
                direction = directions[i],
                amplitude = amplitudes,
                phase = phases[i],
                polarization = polarization[0] * v + polarization[1] * u + polarization[2] * w,
                wavelength = self.wavelength
            )
            for i in range(self.number_of_rays)
        ]

        print("[Laguerre Gaussian Source] :\tDone.")

        return beam
    