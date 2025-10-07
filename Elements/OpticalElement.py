import numpy as np
from Utils import *

class OpticalElement :
    def __init__(self, position, orientation, name):
        self.position = np.array(position)
        self.orientation = normalize(orientation)
        self.name = name
        self.next_elements = set()

    def hit(self, beamlet) :
        raise NotImplementedError("Subclasses must implement this")
    
    def interact(self, beamlet) :
        raise NotImplementedError("Subclasses must implement this.")