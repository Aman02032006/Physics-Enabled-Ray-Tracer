import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cProfile
import pstats

from Sources import *
from Elements import *
from Utils import *
from Tracer import Tracer
from Illustrator import *

source = LaguerreGaussianBeam(
    power = 1.0, 
    radius = 1e-3, 
    wavelength = 633e-9, 
    num_rays = 100000,
    position = [0.0, 0.0, 0.0], 
    direction = [0.0, 0.0, 1.0], 
    polarization = [1.0, 0.0, 0.0],
    p = 2,
    l = 2)

powermeter = PowerMeter(position = [0.0, 0.0, 0.0001], orientation = [0.0, 0.0, 1.0], size = 0.01, name = "Power Meter", pixel_size = 0.00005)

tracer = Tracer(elements = [powermeter])
powermeter.clear()

# source.plot_r_distribution()
beam = source.generate_rays()

for ray in tqdm(beam, desc = "[Simulation] :\tTracing Rays"):
    tracer.trace(ray)

# powermeter.plotPhase()
powermeter.plot(l = source.l, p = source.p)
plt.show()

"""
iris = Iris(position = [0.0, 0.0, 0.1], orientation = [0.0, 0.0 ,1.0], name = "Iris 1", radius = 0.5e-3, size = 0.01)
powermeter = PowerMeter(position = [0.0, 0.0, 0.2], orientation = [0.0, 0.0, 1.0], size = 0.01, name = "Power Meter 1", pixel_size = 0.000025)   

tracer1 = Tracer(elements = [iris])
tracer2 = Tracer(elements = [powermeter])
iris.clear()
powermeter.clear()

beam = source.generate_rays()

for ray in tqdm(beam, desc = "[Simulation] :\tTracing Rays"):
    tracer1.trace(ray)

diffracted_beam = iris.diffract(z = .001)
plt.show()

for ray in tqdm(diffracted_beam, desc = "[Simulation] :\tTracing Rays"):
    tracer2.trace(ray)

powermeter.plot()
plt.show()
"""
"""
# MACH ZEHDNER INTERFEROMETER

beamsplitter1 = BeamSplitter(position = [0.0, 0.0, 0.1], orientation = [1.0, 0.0, -1.0], size = 0.01, name = "Beam Splitter 1", refractive_index = 1.5)
qwp1 = QuarterWavePlate(position = [0.0, 0.0, 0.135], orientation = [0.0, 0.0, 1.0], fast_axis_angle = PI/4, diameter = 0.01, name = "QWP1")
hwp1 = HalfWavePlate(position=[0.0, 0.0, 0.15], orientation=[0.0, 0.0, 1.0], fast_axis_angle = PI/4, diameter=0.01, name="HWP1")
qwp2 = QuarterWavePlate(position=[0.0, 0.0, 0.165], orientation=[0.0, 0.0, 1.0], fast_axis_angle=PI/4,diameter=0.01, name="QWP2")
mirror1 = Mirror(position=[0.0, 0.0, 0.2], orientation = [1.0, 0.0, -1.0], diameter = 0.01, name = "Mirror 1", refractive_index = 1.5)
mirror2 = Mirror(position=[0.1, 0.0, 0.1], orientation = [-1.0, 0.0, 1.0], diameter = 0.01, name = "Mirror 2", refractive_index = 1.5)
beamsplitter2 = BeamSplitter(position = [0.1, 0.0, 0.2], orientation = [-1.0, 0.0, 1.0], size = 0.01, name = "Beam Splitter 2", refractive_index = 1.5)
powermeter1 = PowerMeter(position=[0.1, 0.0, 0.25], orientation = [0.0, 0.0, -1.0], size=0.01, name = "Power Meter 1", pixel_size = 0.0001)
powermeter2 = PowerMeter(position=[0.15, 0.0, 0.2], orientation = [-1.0, 0.0, 0.0], size=0.01, name = "Power Meter 2", pixel_size = 0.0001)

# CreateSetupAssembly(Elements = [beamsplitter1, qwp1, qwp2, hwp1, mirror1, mirror2, beamsplitter2, powermeter1, powermeter2], source = source)
element_list = [beamsplitter1, qwp1, qwp2, hwp1, mirror1, mirror2, beamsplitter2, powermeter1, powermeter2]
tracer = Tracer(elements = element_list)

beam = source.generate_rays()

for ray in tqdm(beam, desc = "[Simulation] :\tTracing Rays"):
    tracer.trace(ray)

powermeter1.plot()
# powermeter2.plot()
plt.show()

# CreateSetupAssembly(source = source, Elements = element_list)
"""
'''
# Lens check

beam = source.generate_rays()

lens1 = PlanoConvexLens(position = [0.0, 0.0, 0.1], orientation = [0.0, 0.0, 1.0], name = "100mm Convex Lens", refractive_index = 1.51509, f_value = 0.1, aperture = 0.025)
lens2 = PlanoConvexLens(position = [0.0, 0.0, 0.4], orientation = [0.0, 0.0, 1.0], name = "200mm Convex Lens", refractive_index = 1.51509, f_value = 0.2, aperture = 0.025, flipped = True)
powermeter = PowerMeter(position = [0.0, 0.0, 0.6], orientation = [0.0, 0.0, 1.0], size = 0.01, name = "Power Meter 1", pixel_size = 0.00001)

tracer = Tracer([*lens1, *lens2, powermeter])

powermeter.clear()

for ray in tqdm(beam, desc = "[Simulation] :\tTracing Rays"):
    tracer.trace(ray)

powermeter.plot()
plt.show()'''
