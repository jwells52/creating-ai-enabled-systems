import os
import cv2
import pandas as pd
from PIL import Image

from easyfsl.datasets import FewShotDataset
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

class HumpbackWhaleDataset(FewShotDataset):
    '''
    EasyFSL FewShotDataset class for loading the Humpback Whale Identification Kaggle dataset.

    Note that FewShotDataset inherits torch.utils.data.Dataset and this class is essentially a PyTorch dataset object.
    '''
    def __init__(self, image_dir: str, labels: pd.DataFrame, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.label_to_id = {label:i for i, label in enumerate(self.labels['Id'].unique())}
        self.id_to_label = {i:label for label, i in self.label_to_id.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = self.labels.iloc[idx]
        image_name, label = item['Image'], self.label_to_id[item['Id']]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        num_channels = image.shape[2]

        image = Image.fromarray(image)
        if self.transform is not None:
            if num_channels != 3:
                gs_transform = transforms.Grayscale(num_output_channels=3)
                image = gs_transform(image)
            image = self.transform(image)

        return image, label

    def get_labels(self):
        return self.labels['Id'].values

def remove_new_whale_class(df: pd.DataFrame):
    '''
    Helper function for removing all images with new_whale class as ID in dataframe
    '''
    _df = df.copy()
    return _df[_df['Id'] != 'new_whale']

def filter_low_occuring_classes(df: pd.DataFrame, threshold:int=10):
    '''
    Helper function for removing classes with number of examples less than specified threshold
    '''
    def class_count(df, label):
        return len(df[df['Id'] == label])

    _df = df.copy()
    _df['class_count'] = _df['Id'].apply(lambda label: class_count(_df, label))
    _df = _df[_df['class_count'] > threshold]

    return _df

def create_loader(
        dataset: FewShotDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int, 
        num_workers: int=2
    ):
    '''
    Helper Function for creating PyTorch dataloader for n-way n-shot learning tasks with n_query size query set

    Arguments:
        dataset (easyfsl.FewShotDataset): Few Shot Dataset (essentially a PyTorch dataset) of dataset wanting to be loaded
        n_way (int): Number of classes in support set
        n_shot (int): Number of examples each class has in the support set
        n_query (int): Number of examples in query set
        n_task (int): Number of tasks
    
    Returns:
        loader (torch.data.utils.Dataloader): PyTorch dataloader for loading meta-learning tasks
    '''
    
    sampler = TaskSampler(
        dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn
    )

    return loader
