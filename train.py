import torch
import argparse
from utilities import train
from preprocess import preprocess
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def main():
    # All the necessary argument
    parser = argparse.ArgumentParser(description="Monai U-Net For 3D image segmentation")
    parser.add_argument('--exp_num', type=int, default=None,
                    help='experiment number')
    parser.add_argument('--dataset', type=str, default=None,
                    help='dataset path. For this create the json file')
    parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
    parser.add_argument('--epochs',type=int, default=4,
                        help='Out channel for multi or single organ segmentation')
    parser.add_argument('--val_interval',type=int, default=2,
                        help='validation interval (default=2)')
    parser.add_argument('--batch_size',type=int, default=1,
                        help='input batch size for training (default = 1)')
    parser.add_argument('--num_worker',type=int, default=1,
                        help='the number of worker threads to use for loading data in parallel (default = 1)')
    parser.add_argument('--num_segments',type=int, default=5,
                        help='how many organ to segment include background')  
    parser.add_argument('--lr',type=float, default=1e-4,
                        help='learning rate')   
    parser.add_argument('--cache',type=bool, default=False,
                        help='cache preload data in memory')   


    args = parser.parse_args()

    # Preprocessing data
    preprocess_data = preprocess()
    preprocess_data.prepare_data(args.dataset,args.exp_num)
    preprocess_data.transform_data()
    data = preprocess_data.dataloader(args.cache,args.batch_size,args.num_worker)

    # prepare model 

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
    loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    dice_metric = DiceMetric(include_background=False,reduction="mean")



    # train the data 
    if args.resume:
        print("resume")
    else:
        train(args.exp_num,model, device, args.epochs, args.val_interval,args.num_segments,data,loss_function,optimizer,dice_metric) 



if __name__ == "__main__":
    main()