import os
import sys
sys.path.append('..')
import json
import base64
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize((256, 512))
])

with tqdm() as bar:
    # do not skip any of the rows, but update the progress bar instead
    images_and_ids = pd.read_csv('images_and_ids.csv', skiprows=lambda x: bar.update(1) and False)



if __name__ == "__main__":
    CLASS_ID   = sys.argv[1]
    NUM_IMAGES = int(sys.argv[2])

    if not os.path.exists(f'encoded/{CLASS_ID}'):
        os.makedirs(f'encoded/{CLASS_ID}')

    image_filenames = images_and_ids[images_and_ids['Id'] == CLASS_ID]['Image'][:NUM_IMAGES]

    for image_fn in image_filenames:
        image_path = f'train/{image_fn}'
        image = Image.open(image_path)

        image = transform(image)
        image.save(image_fn)

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue())
        img_base64_str = img_base64.decode("utf-8")

        with open(f'encoded/{CLASS_ID}/{image_fn.split(".")[0]}', 'w', encoding='utf-8') as f:
            f.write(img_base64_str)