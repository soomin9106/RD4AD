import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
from dataset import SynapseDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test, detection
from torch.nn import functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(_class_):
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 4
    image_size = 512
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = '/home/synapse/' + _class_ + '/Train'
    test_path = '/home/synapse/' + _class_
    ckp_path = '/home/synapse/RD4AD/checkpoints/' + 'wres50_'+_class_+'.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = SynapseDataset(root=test_path, transform=data_transform, phase="Test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = resnet50(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet50(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    save_best_weights = 0
    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))#bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        auroc_sp_max = evaluation(encoder, bn, decoder, test_dataloader, device, _class_)
        print('AUROC {:.3f}'.format(auroc_sp_max))
        if save_best_weights < auroc_sp_max:
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    return auroc_sp_max




if __name__ == '__main__':
    setup_seed(111)
    item_list = ['SMT_220802']
    for i in item_list:
        train(i)

