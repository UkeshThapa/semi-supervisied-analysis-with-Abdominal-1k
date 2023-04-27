import json

from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    ToTensord
    )
from monai.data import DataLoader, Dataset, CacheDataset



class preprocess:

    def prepare_data(self,file_name,exp_num):
        with open('experiment.json') as f:
            data = json.load(f)
        trainingData = data[f'experiment{exp_num}']
        self.train_files, self.val_files = trainingData[:-int(len(trainingData)*0.1)], trainingData[-int(len(trainingData)*0.1):]


    def transform_data(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-200, a_max=200,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),

                Orientationd(keys=["image"], axcodes="RAS"),

                ScaleIntensityRanged(
                    keys=["image"], a_min=-200, a_max=200,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

    def dataloader(self,cache,batch_size,workers):
        if cache:
            train_ds = CacheDataset(
                    data=self.train_files, transform=self.train_transforms,
                    cache_rate=1.0, num_workers=1)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

            val_ds = CacheDataset(
                    data=self.val_files, transform=self.val_transforms, cache_rate=1.0, num_workers=workers)
            val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=1)
            return train_loader, val_loader

        else:
            train_ds = Dataset(data=self.train_files, transform=self.train_transforms)
            train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=workers)

            val_ds = Dataset(data=self.val_files, transform=self.val_transforms)
            val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=workers)

            return train_loader, val_loader