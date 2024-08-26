import numpy as np
import torch
import torchsummary
from PIL import Image

import data_loader
import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def save_to_pt(images, labels, save_path):
    torch.save({'images': torch.tensor(images), 'labels': torch.tensor(labels)}, save_path)

def test_():
    root = 'D:/MyDataSet/CT2MR/PnpAda_release_data'
    _source_train_pth = 'ct_train_list.txt'
    _target_train_pth = 'mr_train_list.txt'
    _source_val_pth = 'ct_val_list.txt'
    _target_val_pth = 'mr_val_list.txt'
    source_train_dataset, target_train_dataset = data_loader.load_data(root, _source_train_pth,
                                                                       _target_train_pth, True)
    source_val_dataset, target_val_dataset = data_loader.load_data(root, _source_val_pth,
                                                                   _target_val_pth, True)
    source_train_datalist = []
    source_train_labellist = []
    target_train_datalist = []
    target_train_labellist = []
    source_val_datalist = []
    source_val_labellist = []
    target_val_datalist = []
    target_val_labellist = []

    for i,(data, label) in enumerate(target_val_dataset):
        data = data.numpy().squeeze()
        label = label.numpy()
        label = np.argmax(label, axis=-1)
        for j in range(data.shape[0]):
            np.save(f'D:/MyDataSet/CT2MR/PnpAda_release_data/pytorch/target_val_data/images_npy/image_{i*data.shape[0]+j}.npy',data[j])
        data = (data + 1) / 2 * 255.0
        data = data.astype(np.uint8)
        for j in range(data.shape[0]):
            img = Image.fromarray(data[j])
            img.save(f'D:/MyDataSet/CT2MR/PnpAda_release_data/pytorch/target_val_data/images/image_{i*data.shape[0]+j}.png')
        for j in range(data.shape[0]):
            np.save(f'D:/MyDataSet/CT2MR/PnpAda_release_data/pytorch/target_val_data/labels/label_{i*data.shape[0]+j}.npy',label[j])


def test2_():
    net = model.MyModel().to(device)
    # discrimitator = net.discriminator_s
    # torchsummary.summary(discrimitator, (1, 256, 256))
    encoder = net.encoder
    torchsummary.summary(encoder, (1, 256, 256))
    # decoder = net.decoder
    # torchsummary.summary(decoder, [(512,32,32),(1,256,256)])
    # pixel_wise_classifier = net.pixel_wise_classifier
    # torchsummary.summary(pixel_wise_classifier, (512,32,32))
    # torchsummary.summary(net, [(1, 256, 256),(1, 256, 256)])

    # for data, labels in target_train_dataset:
    #     print('1')
    #     target_train_datalist.append(data)
    #     target_train_labellist.append(labels)
    # for data, labels in source_val_dataset:
    #     source_val_datalist.append(data)
    #     source_val_labellist.append(labels)
    # for data, labels in target_val_dataset:
    #     print('2')
    #     target_val_datalist.append(data)
    #     target_val_labellist.append(labels)
    # source_train_datalist = np.array(source_train_datalist)
    # source_train_labellist = np.array(source_train_labellist)
    # target_train_datalist = np.array(target_train_datalist)
    # target_train_labellist = np.array(target_train_labellist)
    # source_val_datalist = np.array(source_val_datalist)
    # source_val_labellist = np.array(source_val_labellist)
    # target_val_datalist = np.array(target_val_datalist)
    # target_val_labellist = np.array(target_val_labellist)
    #
    # save_to_pt(source_train_datalist, source_train_labellist, 'data/pytorch/source_train.pt')
    # save_to_pt(target_train_datalist, target_train_labellist, 'data/pytorch/target_train.pt')
    # save_to_pt(source_val_datalist, source_val_labellist, 'data/pytorch/source_val.pt')
    # save_to_pt(target_val_datalist, target_val_labellist, 'data/pytorch/target_val.pt')


def test3_():
    a = input()
    print(a)
test3_()