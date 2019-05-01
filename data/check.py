import glob
from PIL import Image
import os
from multiprocessing import Pool
DIR = 'ILSVRC2012/val/'

def check_cmyk(folder):
    images = glob.glob(os.path.join(DIR,folder,"*.JPEG"))
    for image in images:
        img = Image.open(image)
        if img.mode == 'CMYK':
            img = img.convert("RGB")
            img.save(image)
            return image
        
pool = Pool(processes=6)
print (pool.map(check_cmyk,os.listdir(DIR)))

