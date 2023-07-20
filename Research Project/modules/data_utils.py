import os
import pandas as pd
from PIL import Image
from easyfsl.datasets import FewShotDataset

class HumpbackWhaleDataset(FewShotDataset):
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
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        return self.labels['Id'].values

def remove_new_whale_class(df: pd.DataFrame):
  _df = df.copy()
  return _df[_df['Id'] != 'new_whale']

def filter_low_occuring_classes(df:pd.DataFrame, threshold:int=10):
  def class_count(df, label):
    return len(df[df['Id'] == label])

  _df = df.copy()
  _df['class_count'] = _df['Id'].apply(lambda label: class_count(_df, label))
  _df = _df[_df['class_count'] > threshold]

  return _df


def create_loaders(train_set, valid_set, n_way, n_shot, n_query, n_tasks_training, n_tasks_validation, num_workers=2):
    
  train_sampler = TaskSampler(
      train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_training
  )

  valid_sampler = TaskSampler(
      valid_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_validation
  )

  train_loader = DataLoader(
      train_set,
      batch_sampler=train_sampler,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=train_sampler.episodic_collate_fn
  )

  valid_loader = DataLoader(
      valid_set,
      batch_sampler=valid_sampler,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=valid_sampler.episodic_collate_fn
  )

  return train_loader, valid_loader