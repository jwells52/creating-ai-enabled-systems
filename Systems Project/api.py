'''
FastAPI for model inference using Prototypical Networks
'''
import torch
import base64
import io
import sys
sys.path.append('../Research Paper/')
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from torchvision.transforms import Grayscale
from typing import List, Dict


from modules.train import load_prototypical_network_checkpoint

MODEL_CHECKPOINT_PATH = '/workspaces/creating-ai-enabled-systems/Research Paper/models/checkpoints/prototypical_network_5-way_5-shot_last_epoch'

model = load_prototypical_network_checkpoint(MODEL_CHECKPOINT_PATH, send_to_device=False, map_location=torch.device('cpu'))


class Task(BaseModel):
    '''
    Pydantic Class that validates post request
    '''
    support_set_labels: List[str]
    support_set_images: Dict[str, List[str]]
    query_set_images: List[str]

def load_image_from_bytes(im_b64_str):
    '''
    Load image base64 encoded string into a numpy array
    '''
    # convert it into bytes
    start_index = len('data:image/jpeg;base64,')
    img_bytes = base64.b64decode(im_b64_str[start_index:])

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    img = img.resize((256, 512))

    if len(img.size) != 3:
        grayscale_transform = Grayscale(num_output_channels=3)
        img = grayscale_transform(img)

    # PIL image object to numpy array
    img_arr = np.asarray(img)
    return img_arr

def load_support_set_images_from_bytes(support_set_images_b64_strs, classid_2_idx):
    '''
    Load all base64 encoded images in support set
    '''
    support_set_images, support_set_idxs = [], []
    for class_id, im_b64_list in support_set_images_b64_strs.items():
        support_set_idxs += [classid_2_idx[class_id] for _ in range(len(im_b64_list))]
        support_set_images += [load_image_from_bytes(im_b64) for im_b64 in im_b64_list]
    support_set_images = np.array(support_set_images)
    return support_set_images, support_set_idxs

def load_query_set_images_from_bytes(query_set_images_b64_strs):
    '''
    Load all base64 encoded images in query set
    '''
    query_set_images = []
    for im_b64 in query_set_images_b64_strs:
        query_set_images += [load_image_from_bytes(im_b64)]
    query_set_images = np.array(query_set_images)

    return query_set_images

app = FastAPI()

@app.post("/classify")
async def classify(task: Task):
    '''
    POST method for model inference
    '''
    # Extact support set images/labels and query set images from POST body
    support_set_labels = task.support_set_labels
    support_set_images_b64_strs = task.support_set_images
    query_set_images_b64_strs = task.query_set_images
 
    # These dicts make it easy to select the correct label for a processed query image
    classid_2_idx = {k:i for i, k in enumerate(support_set_labels)}
    idx_2_classid = {i:k for k, i in classid_2_idx.items()}

    # Load all base64 encoded images in POST request
    support_set_images, support_set_idxs = load_support_set_images_from_bytes(support_set_images_b64_strs, classid_2_idx)
    query_set_images = load_query_set_images_from_bytes(query_set_images_b64_strs)

    # Convert all labels and images to torch Tensors
    support_set_idxs = torch.Tensor(support_set_idxs)
    support_set_images = torch.Tensor(support_set_images).permute(0, 3, 1, 2)
    query_set_images = torch.Tensor(query_set_images).permute(0, 3, 1, 2)

    # Pass support set and query set images through Prototypical Network
    model.process_support_set(support_set_images, support_set_idxs)
    predictions = model(query_set_images).detach().data
    prediction_idxs = torch.max(predictions, 1)[1].detach().numpy()
    prediction_classids = [idx_2_classid[idx] for idx in prediction_idxs]

    return {"predicted_classids": prediction_classids}
