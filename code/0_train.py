# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import cfg
import datetime
import torch
import json
import cv2
import torchvision

import utils


import transforms as T
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate

class GjDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.jsons = list(sorted(os.listdir(root)))
        # self.imgs = list(sorted(cfg.img_path))


    def __getitem__(self, idx):
        # load images ad masks
        json_path = os.path.join(self.root, self.jsons[idx])
        with open(json_path, 'r') as f:
            json_dic = json.load(f)
        # print(json_path)
        regions = json_dic['regions']
        img_path = os.path.join(cfg.img_path, json_dic['asset']['name'] + '.jpg')
        # print(json_dic['asset']['name'] + '.jpg')
        img = Image.open(img_path)
        img = img.convert('RGB')
        # img = np.array(img)
        num_objs = len(regions)
        boxes = []
        for ind_box in regions:
            bbox = ind_box['points']
            xmin = bbox[0]['x']
            ymin = bbox[0]['y']
            xmax = bbox[2]['x']
            ymax = bbox[2]['y']
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # print(img.shape)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # print(img,target)
        return img, target
    #
    def __len__(self):
        return len(self.jsons)
'''
获取模型
'''
def get_model_instance_detection(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads. box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
'''
训练函数
'''
def main(data_loader,data_loader_test, model_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_detection(cfg.num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    epoch_resume = 0
#是否继续上一次的训练
    if model_path:
        checkpoint = torch.load(cfg.model_path,  map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch_resume = checkpoint['epoch']
    for epoch in range(epoch_resume, cfg.epoch):
        loss_value = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, cfg.checkpoint_path + '\\model_at_epoch_{}'.format(epoch))


        coco_eval = evaluate(model, data_loader_test, device=device)
        ap_iou_50_95 = coco_eval.coco_eval['bbox'].stats[0]
        ap_iou_50 = coco_eval.coco_eval['bbox'].stats[1]
        ap_iou_75 = coco_eval.coco_eval['bbox'].stats[2]
        with open(cfg.log_path + 'train_log.csv', 'a') as log_file:
            log_file.write(
                '%03d,%0.5f,%0.5f,%0.5f,%0.5f\n' %
                ((epoch),
                 ap_iou_50_95,
                 ap_iou_50,
                 ap_iou_75,
                 loss_value)
            )
    print(datetime.datetime.now().strftime('%F %T'))
    print("That's it!")
'''
加载数据
'''
def load_datasets():
    dataset = GjDataset(cfg.train_path, get_transform(train=True))
    dataset_test = GjDataset(cfg.test_path, get_transform(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    indices_test = torch.randperm(len(dataset_test)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:])
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test[:])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    return data_loader, data_loader_test


if __name__ == "__main__":
    model_path = 1
    data_loader, data_loader_test = load_datasets()
    main(data_loader, data_loader_test, model_path)






