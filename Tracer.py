import numpy as np
from Sources import *
from Utils import *

class Tracer:
    def __init__(self, elements = []):
        self.ElementList = elements
        self.first_element = None

        source = GaussianBeam(
            power = 1.0, 
            radius = 1e-3, 
            wavelength = 633e-9, 
            num_rays = 100,
            position = [0.0, 0.0, 0.0], 
            direction = [0.0, 0.0, 1.0], 
            polarization = [1.0, 0.0, 0.0])
        
        test_rays = source.generate_rays()

        for ray in test_rays:
            self.GetPath(ray)
    
    def GetPath(self, beamlet):
        previous_element = None
        elements_interacted = []

        while beamlet.active:
            min_dist = np.inf
            closest_element = None

            for element in self.ElementList :
                if element in elements_interacted: continue
                dist = element.hit(beamlet)
                
                if dist is not False and dist < min_dist :
                    closest_element = element
                    min_dist = dist
            
            if closest_element is not None :
                # print(f"[Tracer] :\tClosest Element : {closest_element.name}, minimum distance = {min_dist}")
                beamlet.propagate(min_dist)
                closest_element.interact(beamlet)

                if self.first_element is None:
                    self.first_element = closest_element

                if previous_element is not None:
                    previous_element.next_elements.add(closest_element)
                
                previous_element = closest_element
                elements_interacted.append(closest_element)
            else :
                return elements_interacted
    
    def trace(self, beamlet):
        currentElement = None
        distanceToNextElement = False

        while beamlet.active:
            if currentElement is None:
                currentElement = self.first_element
                distanceToNextElement = currentElement.hit(beamlet)
            
            if distanceToNextElement is False:
                return
            # print(f"[Tracer] :\tCurrent Element : {currentElement.name}")

            beamlet.propagate(distanceToNextElement)
            currentElement.interact(beamlet)

            for nextElement in currentElement.next_elements:
                distanceToNextElement = nextElement.hit(beamlet)
                # print(f"[Tracer] :\t{nextElement.name} Hit Checked")
                if distanceToNextElement is not False:
                    currentElement = nextElement
                    break

