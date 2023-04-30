import os
import json
import torch
import monai

import numpy as np
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    ToTensord,
    AsDiscrete,
    Compose
    )
import nibabel as nib
import SimpleITK as sitk
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, CacheDataset



def pre_processing(test_file):
    test_transforms = Compose(
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

    test_ds = Dataset(data=test_file, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)
   
    return test_loader
    


def inference(model,device,test_loader,testingData,dice_metric,num_segments):
    model.eval()
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_segments)])
    post_label = Compose([AsDiscrete(to_onehot=num_segments)])

    with torch.no_grad():
        for i, test_data in enumerate(tqdm(test_loader)):
            org_img = sitk.ReadImage(testingData[i]['label'])
            base_name = os.path.basename(testingData[i]['label'])     
            roi_size = (64, 64, 64)
            sw_batch_size = 4
            test_outputs = sliding_window_inference(
                test_data["image"].to(device), roi_size, sw_batch_size, model
            )
            test_pred_out = [post_pred(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_data["label"].to(device))]
            #             # compute metric for current iteration
            dice_metric(y_pred=test_pred_out, y=test_labels)

            output_image = test_outputs.argmax(dim=1).detach().cpu().numpy().squeeze().astype(np.float32)
            output_image = sitk.GetImageFromArray(output_image)
            output_image.SetOrigin(org_img.GetOrigin())
            output_image.SetDirection(org_img.GetDirection())
            output_image.SetSpacing(org_img.GetSpacing())

            sitk.WriteImage(output_image, f"result/experiment1/{base_name}")
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    with open('metric.txt', 'w') as f:
        f.write(f'Total metric : {metric}')



def main():
    
    num_segments =  5
    # load the image 
    with open('testing_data.json') as f:
        data = json.load(f)
    testingData = data['test']

    # prepare the data 
    test_loader = pre_processing(testingData)

    # initialize the model
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH
        
    ).to(device)

    # load the model you have saved 
    checkpoint = torch.load("result/model_experiment_1.pth.tr")
    model.load_state_dict(checkpoint['state_dict'])

    # dice metric 
    dice_metric = DiceMetric(include_background=False,reduction="mean")

    inference(model,device,test_loader,testingData,dice_metric,num_segments)



if __name__ == "__main__":
    main()