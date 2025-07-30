import numpy as np
from Utils import *

class Beamlet : 
    def __init__(self, position, direction, amplitude, phase, polarization, wavelength) :
        # defining the position and direction of the ray
        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.beam_position = self.position

        self.wavelength = wavelength
        self.k_mag = 2 * PI / wavelength

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
        
        # updating phase
        self.propagation_phase += self.k_mag * distance
