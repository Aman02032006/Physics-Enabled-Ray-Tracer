import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
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
    p = 0,
    l = 1)

frames_dir = os.path.abspath("frames")
os.makedirs(frames_dir, exist_ok=True)

shots = 120

for i in range(shots):
    powermeter = PowerMeter(position=[0.0, 0.0, 0.1 + 9.9 * i / (shots - 1)], orientation = [0.0, 0.0, -1.0], size = 0.01, name = "Power Meter", pixel_size = 0.00005)
    print(f"Propagation Distance : {powermeter.position[2]}")

    tracer = Tracer(elements = [powermeter])
    powermeter.clear()

    beam = source.generate_rays()

    for ray in tqdm(beam, desc = "Tracing rays : "):
        tracer.trace(ray)
    
    powermeter.plot()
    
    figs = [plt.figure(n) for n in plt.get_fignums()]

    if len(figs) >= 1 :
        print("Stitching images") 
        heatmap_file = f"frames/tmp_heatmap_{i:03d}.png"
        cut_file     = f"frames/tmp_cut_{i:03d}.png"
        figs[0].savefig(heatmap_file, dpi=150)
        # figs[1].savefig(cut_file, dpi=150)

        # stitch them together
        im1 = Image.open(heatmap_file)
        # im2 = Image.open(cut_file)

        # resize to same height
        # h = max(im1.height, im2.height)
        # im1 = im1.resize((int(im1.width * h / im1.height), h))
        # im2 = im2.resize((int(im2.width * h / im2.height), h))

        # create new canvas
        # combined = Image.new("RGB", (im1.width + im2.width, h))
        # combined.paste(im1, (0, 0))
        # combined.paste(im2, (im1.width, 0))

        im1.save(os.path.join(frames_dir, f"frame_{i:03d}.png"))
        print("Saved Stitched png")

        # cleanup
        im1.close()
        # im2.close()

        os.remove(heatmap_file)
        # os.remove(cut_file)
    
    plt.close('all')