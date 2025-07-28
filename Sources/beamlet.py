import numpy as np

I = 1j
PI = np.pi
C = 299_792_458

# A beamlet class which defines a single ray.
# It has information about the position of a beamlet, direction, amplitude, phase, and polarization.
# It also calculates at what time approximately the photon reaches a particular optical element.

class Beamlet : 
    def __init__(self, position, direction, amplitude, phase, polarization, wavelength, waist_radius = None, beam_axis = None) :
        # defining the position and direction of the ray
        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))

        self.beam_axis = (
            np.array(beam_axis) / np.linalg.norm(beam_axis)
            if beam_axis is not None
            else None
        )
        self.beam_position = self.position

        self.wavelength = wavelength
        self.k_mag = 2 * PI / wavelength
        self.waist_radius = waist_radius

        self.amplitude = amplitude
        self.propagation_phase = phase
        self.elemental_phase = 0
        self.polarization = np.array(polarization)
        
        self.active = True
        self.timestamp = 0.0
        self.path_length = 0.0
    
    @property
    def E(self):
        return self.amplitude * np.exp(I * (self.propagation_phase + self.elemental_phase)) * self.polarization
    
    def propagate(self, distance) :
        # updating time for which the beam has travelled and the path length
        delta_t = distance / C
        self.timestamp += delta_t
        self.path_length += distance

        # updating the position of the photon
        self.position = self.position + distance * self.direction
        self.beam_position = self.beam_position + distance * self.beam_axis
        
        self.propagation_phase = self.get_phase() % (2 * PI)
    
    def get_phase(self):
        phase = self.k_mag * self.path_length
        return phase

        if self.waist_radius is not None and self.beam_axis is not None :
            z_R = PI * self.waist_radius**2 / self.wavelength

            # wavefront curvature phase
            r_vec = self.position - (self.beam_position + self.beam_axis * ((self.position - self.beam_position) @ self.beam_axis))
            r_squared = np.dot(r_vec, r_vec)

            # curvature radius
            R_z = np.inf if self.path_length < 1e-6 else self.path_length * (1 + (z_R / self.path_length)**2)
            curvature = 0.0 if np.isinf(R_z) else self.k_mag * r_squared / (2 * R_z)

            gouy = np.arctan(self.path_length / z_R)

            phase += curvature - gouy
        
        return phase
