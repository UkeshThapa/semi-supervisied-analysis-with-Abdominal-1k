import os
import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose
)




def train(checkpoint_path, exp_num,model,device,max_epochs,val_interval,num_segments,data,loss_function,optimizer,dice_metric):
    
    # data from the dataloader
    train_loader,val_loader = data
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_segments)])
    post_label = Compose([AsDiscrete(to_onehot=num_segments)])

# resume feature for training
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        best_metric = checkpoint['best_metric']
        best_metric_epoch = checkpoint['epoch']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    else:
        start_epoch = 0
        # variable to store the best metric and best epoch
        best_metric = -1
        best_metric_epoch = -1


    for epoch in range(start_epoch,max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        print(optimizer.state_dict())
        for i,batch_data in enumerate(tqdm(train_loader)):
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        tqdm.write(f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Average Loss: {epoch_loss:.4f}")
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (64, 64, 64)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
        #             # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

        #         # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    state ={
                        'epoch' : best_metric_epoch,
                        'state_dict' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'best_metric' : best_metric

                    }

                    if not os.path.exists('result'):
                        os.makedirs('result')

                    torch.save(state, f".\\result\model_experiment_{exp_num}.pth.tr")
                    
                    print(
                        "\nSaved new best metric model"
                        f"\nBest mean dice: {best_metric:.4f} "
                        f"\nAt epoch: {best_metric_epoch}\n"
                        )
