import random
import torch
import numpy as np

from easyfsl.methods import FewShotClassifier, PrototypicalNetworks
from easyfsl.datasets import FewShotDataset
from easyfsl.utils import evaluate
from tqdm import tqdm
from typing import Callable, List
from torch.optim import Optimizer, SGD, Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import transforms

from modules.data_utils import create_loader

transform = transforms.Compose(
  [
        transforms.Resize((256, 512)),
        transforms.ToTensor()
  ])


random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeatureExtractor(torch.nn.Module):
    '''
    Class for getting outputs of a layer just by calling FeatureExtractor.forward(x) 
    instead of the getting the result in a dictionary when using the create_feature_extractor
    function from torchvision. This is necessary when we want to extract n-dimensional feature maps
    and not a flattened representation.
    '''
    def __init__(self, model, layer_name):
        super().__init__()
        self.model = model
        self.layer_name = layer_name

    def forward(self, x):
        return self.model(x)[self.layer_name]

def load_prototypical_network_checkpoint(savepath, send_to_device=False, map_location=None):
    '''
    Load Prototypical Network from PyTorch checkpoint file.
    '''
    cnn = resnet18()
    cnn.fc = torch.nn.Flatten()

    pt_network = PrototypicalNetworks(cnn)
    if map_location is None:
        pt_network.load_state_dict(
            torch.load(savepath)
        )
    else:
        pt_network.load_state_dict(
            torch.load(savepath, map_location=map_location)
        )

    if send_to_device:
        pt_network = pt_network.to(device)

    return pt_network

def training_epoch(
  model: FewShotClassifier,
  data_loader: DataLoader,
  optimizer: Optimizer,
  loss_fn: Callable,
  use_tqdm: bool = True
):
    '''
    Function for executing a single training epoch for Few Shot Learning.

    One epoch of training contains E episodes. For each episode, there is a support set and query set,
    and the meta-learning will predict the labels in the query set and propogate the error of these predictions to
    the weights of the meta-learner model.

    Arguments:
        model (easyfsl.FewShotClassifier): the meta-learner model being trained
        data_loader (torch.data.utils.DataLoader): Pytorch dataloader for loading episodes in an epoch
        optimizer (torch.optim.Optimizer): Optimization function used for training
        loss_fn (Callable): Loss function that is optimized in training
    
    Keyword Arguments:
        use_tqdm (bool): flag for using tqdm to show training progress (default=True)

    Return:
        np.mean(all_loss) (float): mean of losses occured during training
    '''
  
    all_loss = []
    model.train()

    enumerator = enumerate(data_loader)
    if use_tqdm:
        enumerator = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")

    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in enumerator:
        optimizer.zero_grad()
        model.process_support_set(
            support_images.to(device), support_labels.to(device)
        )
        classification_scores = model(query_images.to(device))

        loss = loss_fn(classification_scores, query_labels.to(device))
        loss.backward()
        optimizer.step()

        all_loss += [loss.item()]

        if use_tqdm:
            enumerator.set_postfix(loss=np.mean(all_loss))

    return np.mean(all_loss)

def train_fsl(
  model: FewShotClassifier,
  train_loader: DataLoader,
  valid_loader: DataLoader,
  optimizer: Optimizer,
  loss_fn: Callable,
  n_epochs: int = 50,
  use_tqdm: bool = True,
  save_model: bool = False,
  save_path: str = '/content/drive/MyDrive/prototypical_network_resnet18'
):  
    '''
    Function for training Few Shot Classifer on all training epochs.

    Arguments:
        model (easyfsl.methods.FewShotClassifier): Few Shot Learning model being trained
        train_loader (torch.data.utils.DataLoader): PyTorch dataloader for loading training epochs
        valid_loader (torch.data.utils.DataLoader): PyTorch dataloader for loading validation epochs
        optimizer (torch.optim.Optimizer): Optimization function used for training
        loss_fn (Callable): Loss function that is optimized during training
    
    Keyword Arguments:
        n_epochs (int): Number of training epochs (default=50)
        use_tqdm (bool): Flag for specifying if tqdm should be used to display training progress (default=True)
        save_model (bool): Flag for specifying if model checkpoints should be saved throughout training (default=False)
        save_path (str): Path that model checkpoints will be saved to ('/content/drive/MyDrive/prototypical_network_resnet18')
    '''

    train_losses = []
    valid_accs = []
    best_valid_acc = 0.0
    for epoch in range(n_epochs):
        if epoch == 0:
            print(f"\nEpoch {epoch+1}")
        else:
            print(f"\nEpoch {epoch+1}", end=' ')
        epoch_loss = training_epoch(model, train_loader, optimizer, loss_fn)
        train_losses += [epoch_loss]
        if valid_loader is not None:
            valid_acc = evaluate(
                model, valid_loader, device=device, tqdm_prefix="Validation"
            )

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_state = model.state_dict()
            
            if save_model:
                print(f'Best performing model, saving state to {save_path}')
                torch.save(best_state, save_path)

            valid_accs += [valid_acc]

        optimizer.step()
  
    if save_path:
        print(f'Saving state of model checkpoint at last epoch to {save_path}_last_epoch')
        torch.save(model.state_dict(), save_path+'_last_epoch')

    return train_losses, valid_accs

def train_network(
    network: FewShotClassifier,
    train_dataset: FewShotDataset,
    n_ways:List[int],
    n_shots: List[int],
    n_query: int,
    n_tasks_per_epoch: int,
    checkpoint_path: str,
    n_workers: int=12,
    feature_maps: bool=False,
    return_layer: str='layer4.1.bn2',
    learning_rate: float=1e-2,
    n_epochs: int=10
  ):

    '''
    Function for training a Few Shot Learning network on various ways and shots. 

    If n_ways = [5, 10] and n_shots = [5, 20] then the following n-ways k-shots learning networks are trained:
        - 5-way 5-shot
        - 5-way 20-shot
        - 10-way 5-shot
        - 20-way 20-shot
    '''
    losses = dict()

    for n_way in n_ways:
        losses[n_way] = dict()
        for n_shot in n_shots:
            train_loader = create_loader(train_dataset, n_way, n_shot, n_query, n_tasks_per_epoch, num_workers=n_workers)

            resnet = resnet18(weights='DEFAULT')
            resnet.fc = torch.nn.Flatten()

            if feature_maps:
                resnet_extractor = create_feature_extractor(resnet, return_nodes=[return_layer])
                feature_extractor = FeatureExtractor(resnet_extractor, return_layer).to(device)
                fsl_network = network(feature_extractor, feature_dimension=512).to(device)

            else:
                feature_extractor = resnet
                fsl_network = network(feature_extractor).to(device)

            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = Adam(fsl_network.parameters(), lr=learning_rate)

            print(f'Training network under {n_way}-way {n_shot}-shot')
            train_losses, _ = train_fsl(
                fsl_network,
                train_loader, None,
                optimizer, loss_fn, n_epochs=n_epochs,
                save_model=True, save_path=f'{checkpoint_path}_{n_way}-way_{n_shot}-shot'
            )

            losses[n_way][n_way] = train_losses