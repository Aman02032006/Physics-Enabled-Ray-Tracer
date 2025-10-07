import numpy as np

I = 1j
PI = np.pi
Ep0 = 8.854187817e-12
C = 299792458
EPSILON = 1e-6

def vec_add(a, b):
    return np.array([a[0] + b[0], a[1] + b[1], a[2] + b[2]])

def vec_sub(a, b):
    return np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

def norm(a):
    return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def scale(a, s):
    return np.array([a[0] * s, a[1] * s, a[2] * s])

def normalize(a) :
    len = norm(a)
    return (a / len) if len > 1e-6 else (a / 1e-6)

def propagate_angular_spectrum(E, dx, dy, k0, z, n = 1.0, pad_factor=2):
    """
    E:           2D complex field at z=0 (shape: Ny x Nx)
    dx, dy:      sampling steps [m]
    wavelength:  vacuum wavelength [m]
    z:           propagation distance [m]
    n:           refractive index
    pad_factor:  zero-padding factor to reduce circular wrap-around
    """
    
    k  = n * k0

    Ny, Nx = E.shape
    Ny2, Nx2 = int(pad_factor * Ny), int(pad_factor * Nx)

    # zero pad (centered)
    pad_y = (Ny2 - Ny) // 2
    pad_x = (Nx2 - Nx) // 2
    E_pad = np.pad(E, ((pad_y, Ny2 - Ny - pad_y), (pad_x, Nx2 - Nx - pad_x)), mode='constant')

    # spatial frequencies
    fx = np.fft.fftfreq(Nx2, d=dx)   # [1/m]
    fy = np.fft.fftfreq(Ny2, d=dy)   # [1/m]
    FX, FY = np.meshgrid(fx, fy, indexing='xy')

    KX = 2 * PI * FX
    KY = 2 * PI * FY

    # kz = +sqrt(k^2 - kx^2 - ky^2), allow complex for evanescent components
    KZ = np.sqrt((k**2) - (KX**2 + KY**2) + 0j)

    H = np.exp(1j * KZ * z)          # transfer function

    Efx = np.fft.fft2(E_pad)
    Ez_pad = np.fft.ifft2(Efx * H)

    # crop back to original size
    Ez = Ez_pad[pad_y:pad_y+Ny, pad_x:pad_x+Nx]
    return Ez

def field_phase_gradients(E, dx, dy):
    """
    Returns phase gradients (dphi/dx, dphi/dy) [rad/m] using unwrap+central diffs.
    """
    # unwrap along each axis to avoid 2π jumps
    phi = np.angle(E)
    phi = np.unwrap(np.unwrap(phi, axis=1), axis=0)

    dphi_dy = np.zeros_like(phi)
    dphi_dx = np.zeros_like(phi)

    # central differences, one-pixel interior stencil
    dphi_dx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*dx)
    dphi_dy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dy)

    # fallback to forward/backward at borders
    dphi_dx[:, 0]  = (phi[:, 1] - phi[:, 0]) / dx
    dphi_dx[:, -1] = (phi[:, -1] - phi[:, -2]) / dx
    dphi_dy[0, :]  = (phi[1, :] - phi[0, :]) / dy
    dphi_dy[-1, :] = (phi[-1, :] - phi[-2, :]) / dy

    return dphi_dx, dphi_dy