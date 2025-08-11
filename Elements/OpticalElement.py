import numpy as np

class OpticalElement :
    def __init__(self, position, orientation, name):
        self.position = np.array(position)
        self.orientation = np.array(orientation) / np.linalg.norm(np.array(orientation))
        self.name = name
        self.next_elements = set()

    def hit(self, beamlet) :
        raise NotImplementedError("Subclasses must implement this")
    
    def interact(self, beamlet) :
        raise NotImplementedError("Subclasses must implement this.")