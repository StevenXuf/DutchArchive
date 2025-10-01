import os
import glob
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils import get_default_config

class ArchiveDataset(Dataset):
    def __init__(self, image_folder, anno_folder):
        # Load all CSV files in the folder
        self.image_folder = image_folder
        self.anno_folder = anno_folder
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))
        dataframes = []

        for file in all_files:
            df = pd.read_csv(file)

            # Drop rows with missing or empty 'image' or 'description'
            df = df.dropna(subset=['photo_id', 'description'])
            df = df[(df['photo_id'].astype(str).str.strip() != '') & 
                    (df['description'].astype(str).str.strip() != '')]

            dataframes.append(df)

        # Combine all CSVs into one DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['photo_id']
        image_folder = os.path.join(self.image_folder, image_id[:2], image_id[2:4], f'{image_id}'.jpg)
        pil_image = Image.open(image_folder).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
        description = row['description']

        # Return as dictionary (can be customized for your use case)
        return {
            'image': pil_image,
            'image_id': image_id,
            'caption': description
        }
    
def get_archive_loader(image_folder, anno_folder, batch_size=16, num_workers=0, shuffle=True, transform=None):
    collate_fn = lambda batch:{
        'image': torch.stack([transform(item['image']) if transform is not None else item['image'] for item in batch]),
        'pil_image': [item['image'] for item in batch],
        'image_id': [item['image_id'] for item in batch],
        'caption': [item['caption'] for item in batch]
    }

    dataset = ArchiveDataset(image_folder=image_folder, anno_folder=anno_folder)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        pin_memory=True,
                        num_workers=num_workers,
                        collate_fn=collate_fn
    )
    return loader


if __name__ == "__main__":
    cfg = get_default_config()
    image_folder = cfg['IMAGE_FOLDER']
    anno_folder = cfg['ANNOTATION_FOLDER']
    batch_size = cfg['BATCH_SIZE']
    data_loader = get_archive_loader(image_folder=image_folder, anno_folder=anno_folder, batch_size=batch_size)

    for batch in data_loader:
        image = batch['image']
        print(image.size())
        print
        break
