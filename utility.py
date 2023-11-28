import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import json
import numpy as np

import matplotlib.pyplot as plt

from collections import namedtuple

## From https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

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
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       19 , 'void'           , 7       , False        , True         , (  0,  0,142) ),
]




class CityscapeSegTemp(Dataset):
    def __init__(self, root_dir, device, split='train' ):
        self.root_dir = root_dir
        self.split = split
        self.img_dir = os.path.join(root_dir, f'leftImg8bit/{split}')
        self.mask_dir = os.path.join(root_dir, f'gtFine/{split}')
        self.json_dir = os.path.join(root_dir, f'vehicle/{split}')
        self.img_list = glob.glob(f'{self.img_dir}/*/*_leftImg8bit.png')
        self.device = device

        self.id_to_trainId = {label.id: label.trainId for label in labels}


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        device = self.device
        img_path = self.img_list[idx]
        city, filename = os.path.split(img_path)
        city = os.path.basename(city)
        base_filename = filename.split('_leftImg8bit.png')[0]
        json_path = os.path.join(self.json_dir, city, f'{base_filename}_vehicle.json')
        mask_path = os.path.join(self.mask_dir, city, f'{base_filename}_gtFine_labelIds.png')

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Resize image and mask to 300x150 while maintaining aspect ratio
        img = img.resize((128, 64), Image.LANCZOS)
        mask = mask.resize((128, 64), Image.NEAREST)  # Use NEAREST for segmentation masks to avoid introducing new labels

        mask_array = np.array(mask)

        remap = np.vectorize(self.id_to_trainId.get)
        mask_remapped_array = remap(mask_array)


        # Load json and extract temperature value
        with open(json_path, 'r') as f:
            data = json.load(f)
        temperature = data['outsideTemperature']

        # Convert to PyTorch tensors
        img_tensor = transforms.functional.to_tensor(img)
        # mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
        mask_tensor = torch.from_numpy(mask_remapped_array).long()

        temperature_tensor = torch.tensor(temperature, dtype=torch.float32)


        return img_tensor.to(device), (mask_tensor.to(device), temperature_tensor.to(device))


from matplotlib.colors import ListedColormap

colors = [label.color for label in labels if label.trainId != -1 and label.trainId != 255]
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
cityscapes_cmap = ListedColormap(colors)



def plot_samples(data_loader, num_samples=4):
    # Fetch a batch of samples
    images, (masks, temperatures) = next(iter(data_loader))
    
    fig, axs = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        img = images[i].detach().cpu().numpy().transpose((1, 2, 0))
        mask = masks[i].detach().cpu().numpy()
        temp = temperatures[i].detach().cpu().item()
        
        axs[0, i].imshow(img)
        axs[0, i].set_title(f'Temperature: {temp}°C')
        axs[0, i].axis('off')
        
        axs[1, i].imshow(mask, cmap='tab20')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()


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
