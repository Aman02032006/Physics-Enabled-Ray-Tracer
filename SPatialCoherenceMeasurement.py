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

source1 = GaussianBeam(
    power = 1.0, 
    radius = 1e-3, 
    wavelength = 633e-9, 
    num_rays = 100000,
    position = [0.001, 0.0, 0.0], 
    direction = [0.0, 0.0, 1.0], 
    polarization = [1.0, 0.0, 0.0])

source2 = GaussianBeam(
    power = 1.0, 
    radius = 1e-3, 
    wavelength = 633e-9, 
    num_rays = 100000,
    position = [-0.001, 0.0, 0.0], 
    direction = [0.0, 0.0, 1.0], 
    polarization = [1.0, 0.0, 0.0])

beamsplitter = BeamSplitter(position = [0.0, 0.0, 0.1], orientation = [1.0, 0.0, -1.0], size = 0.025, name = "Beam Splitter", refractive_index = 1.5)
lens1 = PlanoConvexLens(position = [0.04, 0.0, 0.1], orientation = [1.0, 0.0, 0.0], name = "100mm Plano-Convex Lens", refractive_index = 1.51509, f_value = 0.1, aperture = 0.025)
mirror1 = Mirror(position=[0.14, 0.0, 0.1], orientation = [-1.0, 0.0, 0.0], diameter = 0.025, name = "Mirror 1", refractive_index = 1.5)
mirror2 = Mirror(position=[0.0, 0.0, 0.24], orientation = [0.0, 0.0, -1.0], diameter = 0.025, name = "Mirror 2", refractive_index = 1.5)
powermeter = PowerMeter(position=[-0.1, 0.0, 0.1], orientation = [1.0, 0.0, 0.0], size = 0.01, name = "Power Meter", pixel_size = 0.00005)

elements = [beamsplitter, *lens1, mirror1, mirror2, powermeter]

CreateSetupAssembly(source=source1, Elements=[beamsplitter, lens1, mirror1, mirror2, powermeter])

tracer = Tracer(elements = elements)
powermeter.clear()

beam1 = source1.generate_rays()
beam2 = source2.generate_rays()

for ray in tqdm(beam1, desc = "[Simulation] :\tTracing Rays"):
    tracer.trace(ray)

for ray in tqdm(beam2, desc = "[Simulation] :\tTracing Rays"):
    tracer.trace(ray)

powermeter.plot()
plt.show()