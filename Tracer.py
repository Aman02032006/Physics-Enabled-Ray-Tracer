import numpy as np
from Utils import *

class Tracer:
    def __init__(self, elements = []):
        self.ElementList = elements
    
    def trace(self, beamlet):
        interacted_elements = []
        while beamlet.active:
            min_dist = np.inf
            closest_element = None

            for element in self.ElementList :
                if element in interacted_elements : continue
                # print(f"[Tracer] :\t{element.name} Hit Checked")
                dist = element.hit(beamlet)
                
                if dist is not False and dist < min_dist :
                    closest_element = element
                    min_dist = dist
            
            if closest_element is not None :
                # print(f"[Tracer] :\tClosest Element : {closest_element.name}, minimum distance = {min_dist}")
                beamlet.propagate(min_dist)
                closest_element.interact(beamlet)
                interacted_elements.append(closest_element)
            else :
                return interacted_elements
        
        return interacted_elements
            