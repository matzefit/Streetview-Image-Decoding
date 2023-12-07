import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt

from collections import namedtuple



## From https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
## modified cat id 

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )





labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      10 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      10 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      10 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      10 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      10 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      10 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      10 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      1, 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      10 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        3 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      10 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      3 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      3 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        10 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,        10 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        10 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        10 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        4 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        5 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        6 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        6 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        7 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       8 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       8 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,       8 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,       8 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,      10 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       9 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       9 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      10 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]



colors = [
          (128, 64,128) , # road (flat)
          (244, 35,232) , # sidewalk
          ( 70, 70, 70) , # building
          (102,102,156) , # construction
        #   (153,153,153) , # object
          (107,142, 35) , # vegetation
        #   (152,251,152) , # terrain
          ( 70,130,180) , # sky
          (220, 20, 60) , # human
          (  0,  0,142) , # car
          (  0,  0, 70) , # truck / bus
          (  0,  20,230) , # motorcycle / bicycle
          (  0,  0,  0)   # void
          ]




from matplotlib.colors import ListedColormap
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
cityscapes_cmap = ListedColormap(colors)



class CityscapeSegTemp(Dataset):
    def __init__(self, root_dir, device, split='train', transform=None, temp_scale=None ):
        self.root_dir = root_dir
        self.split = split
        self.img_dir = os.path.join(root_dir, f'leftImg8bit/{split}')
        self.mask_dir = os.path.join(root_dir, f'gtFine/{split}')
        self.json_dir = os.path.join(root_dir, f'vehicle/{split}')
        self.img_list = glob.glob(f'{self.img_dir}/*/*_leftImg8bit.png')
        self.device = device
        self.transform = transform
        self.temp_scale = temp_scale


        self.id_to_trainId = {label.id: label.trainId for label in labels if label.trainId != 255}



    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        city, filename = os.path.split(img_path)
        city = os.path.basename(city)
        base_filename = filename.split('_leftImg8bit.png')[0]
        json_path = os.path.join(self.json_dir, city, f'{base_filename}_vehicle.json')
        mask_path = os.path.join(self.mask_dir, city, f'{base_filename}_gtFine_labelIds.png')

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img = img.resize((128, 64), Image.LANCZOS)
        mask = mask.resize((128, 64), Image.NEAREST)  

        if self.transform:
            img = self.transform(img)

        mask_array = np.array(mask, dtype=np.int32)
        mask_remapped_array = np.vectorize(self.id_to_trainId.get, otypes=[np.int32])(mask_array, 10)

        # mask_remapped = Image.fromarray(mask_remapped_array.astype(np.uint8))
        # mask_resized = mask_remapped.resize((128, 64), Image.NEAREST)


        # Load json and extract temperature value
        with open(json_path, 'r') as f:
            data = json.load(f)
        temperature = data['outsideTemperature']

        if self.temp_scale is not None:
            temp_min, temp_max = self.temp_scale
            temperature = (temperature - temp_min) / (temp_max - temp_min)

        # Convert the image to tensor if not already done
        if not isinstance(img, torch.Tensor):
            img_tensor = transforms.functional.to_tensor(img)
        else:
            img_tensor = img


        # img_tensor = transforms.functional.to_tensor(img)
        mask_tensor = torch.from_numpy(np.array(mask_remapped_array)).long()
        temperature_tensor = torch.tensor(temperature, dtype=torch.float32)


        return img_tensor.to(self.device), (mask_tensor.to(self.device), temperature_tensor.to(self.device))




def get_min_max(root_dir, split='train'):
    json_dir = os.path.join(root_dir, f'vehicle/{split}')
    json_paths = glob.glob(os.path.join(json_dir, '**', '*_vehicle.json'), recursive=True)
    
    min_temp = float('inf')
    max_temp = float('-inf')

    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            temperature = data['outsideTemperature']
            min_temp = min(min_temp, temperature)
            max_temp = max(max_temp, temperature)

    return min_temp, max_temp


def unscale_temp(scaled_temp, min_temp, max_temp):
    return scaled_temp * (max_temp - min_temp) + min_temp


## -------------------------
## ---- calculate ious -----
## -------------------------


def calculate_iou(preds, labels, num_classes):
    iou_per_class = []
    preds = torch.argmax(preds, dim=1).view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        preds_cls = preds == cls
        labels_cls = labels == cls
        intersection = (preds_cls & labels_cls).sum().item()
        union = (preds_cls | labels_cls).sum().item()

        if union == 0:
            # Exclude from mean IoU if there are no ground truth pixels for this class in the batch
            continue

        iou = intersection / union
        iou_per_class.append(iou)

    return iou_per_class



import torch
import torch.nn.functional as F

## -------------------------
## ---- Train Function -----
## -------------------------

def train(epoch, model,train_loader, optimizer, log_interval, num_classes, train_loss, train_epoch_iou, train_mask_loss , train_temp_loss):
    model.train()
    total_mask_loss = 0.0
    total_temp_loss = 0.0
    train_running_loss = 0.0

    train_iou = []

    for batch_idx, (images, (masks, temperatures)) in enumerate(train_loader):
        images, masks, temperatures = images, masks, temperatures

        optimizer.zero_grad()
        output_masks, output_temps = model(images)

        mask_loss = F.cross_entropy(output_masks, masks.long())  
        total_mask_loss += mask_loss.item()

        temp_loss = F.mse_loss(output_temps.squeeze(), temperatures.float()) 
        total_temp_loss += temp_loss.item()
        
        
        # Combine losses and backpropagate
        total_loss = mask_loss + temp_loss
        total_loss.backward()
        optimizer.step()

        train_running_loss += total_loss.item()  # Update running loss

        iou_batch = calculate_iou(output_masks, masks, num_classes)
        train_iou.extend(iou_batch)

        # Logging
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss.item():.3f}\t'
                  f'Mask Loss: {mask_loss.item():.3f}, Temp Loss: {temp_loss.item():.3f}')

    # Calculate average training loss for the epoch
    # train_epoch_loss = train_running_loss / len(train_loader.dataset)

    epoch_mask_loss = total_mask_loss / len(train_loader) 
    epoch_temp_loss = total_temp_loss / len(train_loader)

    train_epoch_loss = train_running_loss / len(train_loader)
    mean_iou_train = sum(train_iou) / len(train_iou)
    
    print(f'Train Epoch Loss: {train_epoch_loss:.3f} ------ Mask Loss: {epoch_mask_loss:.3f} ---- Temp Loss: {epoch_temp_loss:.3f} ---- Mean IOU: {mean_iou_train:.3f}')

    train_loss.append(train_epoch_loss)
    train_epoch_iou.append(mean_iou_train)
    train_mask_loss.append(epoch_mask_loss)
    train_temp_loss.append(epoch_temp_loss)


## ----------------------------
## ---- Evaluate Function -----
## ----------------------------

def evaluate(model, data_loader, device, num_classes, val_loss, val_epoch_iou, val_mask_loss, val_temp_loss):
    model.to(device)
    model.eval()
    total_mask_loss = 0.0
    total_temp_loss = 0.0
    iou = []

    with torch.no_grad():  # Disable gradient calculation during validation
        for images, (masks, temperatures) in data_loader:
            images, masks, temperatures = images.to(device), masks.to(device), temperatures.to(device)
            output_masks, output_temps = model(images)

            # Calculate and accumulate mask loss
            mask_loss = F.cross_entropy(output_masks, masks.long(), reduction='sum').item()
            total_mask_loss += mask_loss

            # Ensure temperature dimensions are consistent
            output_temps = output_temps.squeeze()
            if output_temps.ndim == 0:  # Output is a scalar, add a batch dimension
                output_temps = output_temps.unsqueeze(0)
            temperatures = temperatures.float().view_as(output_temps)

            # Calculate and accumulate temperature loss
            temp_loss = F.mse_loss(output_temps, temperatures, reduction='sum').item()
            total_temp_loss += temp_loss

            # Calculate and store IoU for each batch
            iou_batch = calculate_iou(output_masks, masks, num_classes)
            iou.extend(iou_batch)

        # Normalize the mask loss per pixel and calculate average losses
        total_mask_loss /= len(data_loader.dataset) * masks.size(-2) * masks.size(-1)
        total_temp_loss /= len(data_loader.dataset)

        # Combine losses with consistent weighting
        epoch_loss = total_mask_loss + total_temp_loss 

        # Calculate mean IoU over all batches
        mean_iou_val = sum(iou) / len(iou)

        val_loss.append(epoch_loss)
        val_epoch_iou.append(mean_iou_val)
        val_mask_loss.append(total_mask_loss)
        val_temp_loss.append(total_temp_loss)

        print(f'Valid total loss: {epoch_loss:.3f} ------ Mask Loss: {total_mask_loss:.3f} ---- Temp Loss: {total_temp_loss:.3f} ---- Mean IOU: {mean_iou_val:.3f}')

        return mean_iou_val


## ----------------------------
## -------- plotting ----------
## ----------------------------




def plot_class_colors(classes, cmap):

    fig, ax = plt.subplots(figsize=(8, 1))
    cols = cmap.colors
    n = len(classes)
    
    # Check if the number of classes matches the number of colors
    if n != len(cols):
        raise ValueError("Number of classes and colormap entries do not match.")
    
    # Create an array with class indices to map colors
    class_indices = list(range(n))
    
    # Create a color bar with the class indices and colormap
    cb = ax.imshow([class_indices], cmap=cmap)
    
    # Set the number of ticks and labels on the color bar
    cbar = plt.colorbar(cb, orientation="horizontal", ticks=class_indices, aspect=50)
    cbar.ax.set_xticklabels(classes, rotation='vertical')
    
    # Remove axis
    ax.axis('off')
    
    plt.show()


def plot_samples(data_loader, num_samples=4, unnormalize=False, mean=None, std=None, temp_scale=None):
    # Fetch a batch of samples
    images, (masks, temperatures) = next(iter(data_loader))
    
    fig, axs = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        img = images[i].detach().cpu().numpy().transpose((1, 2, 0))
        mask = masks[i].detach().cpu().numpy()
        temp = temperatures[i].detach().cpu().item()


        if unnormalize and mean is not None and std is not None:
            img = img * std + mean  
            img = np.clip(img, 0, 1) 

        if temp_scale is not None:
            temp_min, temp_max = temp_scale
            temp = unscale_temp(temp, temp_min, temp_max)
            
        
        axs[0, i].imshow(img)
        axs[0, i].set_title(f'Temperature: {temp:.2f}°C')
        axs[0, i].axis('off')

        mask[mask == 255] = 10

        axs[1, i].imshow(mask, cmap=cityscapes_cmap)
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_predictions(model, data_loader, device, num_samples=5, unnormalize=False, mean=None, std=None, temp_scale = None):
    # Fetch a batch of samples
    images, (true_masks, true_temperatures) = next(iter(data_loader))
    images = images.to(device)
    
    # Get the model predictions
    model.eval()
    with torch.no_grad():
        output_masks, output_temps = model(images)
        pred_masks = output_masks.argmax(dim=1).cpu()
        pred_temps = output_temps.cpu()

    # Set up the plot
    fig, axs = plt.subplots(3, num_samples, figsize=(15, 9))  # 3 rows for images, true masks, and predicted masks
    
    for i in range(num_samples):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        true_mask = true_masks[i].cpu().numpy()
        true_temp = true_temperatures[i].cpu().numpy()
        pred_mask = pred_masks[i].numpy()
        pred_temp = pred_temps[i].item()
        
        # Unnormalize the image if specified
        if unnormalize and mean is not None and std is not None:
            img = img * std + mean
            img = np.clip(img, 0, 1)

        if temp_scale is not None:
            temp_min, temp_max = temp_scale
            true_temp = unscale_temp(true_temp, temp_min, temp_max)
            pred_temp = unscale_temp(pred_temp, temp_min, temp_max)
            
        
        # Original image
        axs[0, i].imshow(img)
        axs[0, i].set_title(f'Original Image')
        axs[0, i].axis('off')

        true_mask[true_mask == 255] = 10

        # True mask
        axs[1, i].imshow(true_mask, cmap=cityscapes_cmap)
        axs[1, i].set_title(f'True Mask Temp: {true_temp:.2f}°C')
        axs[1, i].axis('off')

        # Predicted mask with temperature as title
        axs[2, i].imshow(pred_mask, cmap=cityscapes_cmap)
        axs[2, i].set_title(f'Pred Mask Temp: {pred_temp:.2f}°C')
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_metrics(model, data_loader, device, num_classes, temp_scale= None):
    all_preds = []
    all_trues = []
    true_temps = []
    pred_temps = []
    
    model.eval()
    with torch.no_grad():
        for images, (masks, temperatures) in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            temperatures = temperatures.to(device)

            output_masks, output_temps = model(images)
            preds = output_masks.argmax(dim=1).cpu().numpy()
            temps = output_temps.squeeze().cpu().numpy()

            if temp_scale is not None:
                temp_min, temp_max = temp_scale
                temperatures = unscale_temp(temperatures, temp_min, temp_max)
                temps = unscale_temp(temps, temp_min, temp_max)

            all_trues.extend(masks.cpu().numpy().flatten())
            all_preds.extend(preds.flatten())
            true_temps.extend(temperatures.cpu().numpy())
            pred_temps.extend(temps)

    # Compute the confusion matrix
    cm = confusion_matrix(all_trues, all_preds, labels=list(range(num_classes)))

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Fit a line to the scatter plot data
    slope, intercept, r_value, _, _ = stats.linregress(true_temps, pred_temps)
    line_fit = [slope * x + intercept for x in true_temps]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Confusion Matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axs[0], xticklabels=True, yticklabels=True)
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('True')
    axs[0].set_title('Normalized Confusion Matrix')

   

    # Scatter Plot with Line Fit and R²
    axs[1].scatter(true_temps, pred_temps, alpha=0.5)
    axs[1].plot(true_temps, line_fit, 'r')
    axs[1].set_xlabel('True Temperatures')
    axs[1].set_ylabel('Predicted Temperatures')
    axs[1].set_title(f'True vs Predicted Temperatures (R²: {r_value**2:.2f})')

    plt.tight_layout()
    plt.show()