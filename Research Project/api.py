import base64
import io
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torchvision.models import resnet18
from easyfsl.methods import PrototypicalNetworks
from typing import List



print('Loading Prototypical Network...')
cnn = resnet18()
cnn.fc = torch.nn.Flatten()

model = PrototypicalNetworks(cnn).to('cpu')
model.load_state_dict(
  torch.load('/workspaces/creating-ai-enabled-systems/Research Project/models/checkpoints/prototypical_network_5-way_5-shot_last_epoch',map_location=torch.device('cpu'))
)
print('finished!')

app = FastAPI()

class Task(BaseModel):
    '''
    Pydantic Class that validates post request
    '''
    support_set_labels: List[str]
    support_set_images: List[List[str]]
    query_set_images: List[str]


@app.post("/classify")
async def classify(task: Task):
    '''
    POST method for model inference
    '''
    
    support_set_labels = task.support_set_labels
    support_set_images_b64_strs = task.support_set_images
    query_set_images_b64_strs = task.query_set_images

    support_set_images = []
    query_set_images = []
    print(support_set_labels)
    print(support_set_images)
    print(query_set_images)
    
    for im_b64 in support_set_images_b64_strs:
    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)      
    print('img shape', img_arr.shape)

    return {"message": "Hello World"}