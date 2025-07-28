import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from Elements.OpticalElement import OpticalElement
from tqdm import tqdm
from Utils import *

class PhaseMeter(OpticalElement):
    def __init__(self, position, orientation, name, size, pixel_size = 0.0001):
        super().__init__(position, orientation, name)
        self.size = size
        self.pixel_size = pixel_size
        self.collected_beamlets = []

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

        u, v, _ = self.local_frame

        intersection_point = beamlet.position + t * beamlet.direction
        x_point = np.dot((intersection_point - self.position), v)
        y_point = np.dot((intersection_point - self.position), u)

        if abs(x_point) > self.size / 2 or abs(y_point) > self.size / 2 : return False

        return t
    
    def interact(self, beamlet):
        self.collected_beamlets.append(beamlet)
        beamlet.active = False
    
    def plot(self):
        if not self.collected_beamlets:
            print("No beamlets collected.")
            return

        u, v, _ = self.local_frame
        x_list = []
        y_list = []
        extracted_phase_list = []
        calculated_phase_list = []

        for beamlet in self.collected_beamlets:
            dx = beamlet.position - self.position
            x_rel = np.dot(dx, v)
            y_rel = np.dot(dx, u)
            
            x_list.append(x_rel)
            y_list.append(y_rel)

            # print(f"[{self.name}] :\tElectric Field = {beamlet.E}")

            Ex = np.dot(beamlet.E, v)
            Ey = np.dot(beamlet.E, u)
            phi = np.angle(Ex, deg = False)
            extracted_phase_list.append(phi if phi > 0 else phi + 2 * PI)  # Wrap phase between 0 and 2π
            calculated_phase_list.append(beamlet.propagation_phase)
        
        print(f"Average total phase = {np.mean(extracted_phase_list) / PI} PI")
        print(f"Average propagation phase = {np.mean(calculated_phase_list) / PI} PI")
        
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(x_list, y_list, c=calculated_phase_list, cmap='twilight', s=2, vmin = 0.0, vmax = 2 * PI)
        plt.colorbar(scatter, label='Phase (radians)')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.xlim(-self.size / 2, self.size / 2)
        plt.ylim(-self.size / 2, self.size / 2)
        plt.title("Phase Map at PhaseMeter")
        plt.gca().set_aspect('equal')
        plt.show()