import random
import torch
import numpy as np

from easyfsl.methods import FewShotClassifier
from easyfsl.utils import evaluate
from tqdm import tqdm
from typing import Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def training_epoch(
  model: FewShotClassifier,
  data_loader: DataLoader,
  optimizer: Optimizer,
  loss_fn: Callable,
  use_tqdm: bool = True
):
  
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
  save_path: str = '/content/drive/MyDrive/prototypical_network_resnet12'
):

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
              torch.save(model.state_dict(), save_path)

        valid_accs += [valid_acc]

      optimizer.step()
  
  if save_path:
    print(f'Saving state of model checkpoint at last epoch to {save_path}_last_epoch')
    torch.save(model.state_dict(), save_path+'_last_epoch')
  
  return train_losses, valid_accs
