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
