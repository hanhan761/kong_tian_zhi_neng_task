# camera-ready
import torch
import torch.utils.data
from PIL import Image
import numpy as np
import cv2
import os

train_dirs = ["seq1/", "seq2/", "seq3/", "seq4/", "seq5/", 
              "seq6/", "seq7/", "seq8/", "seq9/", "seq10/",
              "seq11/", "seq12/", "seq13/", "seq14/", "seq15/",
              "seq31/", "seq32/", "seq33/", "seq34/", "seq35/"]
val_dirs = ["seq16/", "seq17/", "seq18/","seq19/",
            "seq20/", "seq36/", "seq37/"]
test_dirs = ["seq21/", "seq22/", "seq23/", "seq24/", "seq25/",
             "seq26/", "seq27/", "seq28/", "seq29/", "seq30/",
             "seq38/", "seq39/", "seq40/", "seq41/", "seq42/" ]

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, uavid_data_path, uavid_meta_path, transform=None):
        self.img_dir = uavid_data_path + "/train/"
        self.label_dir = uavid_meta_path + "/labelimg/train/"
        self.transform = transform

        self.img_h = 2160
        self.img_w = 3840

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for train_dir in train_dirs:
            train_img_dir_path = self.img_dir + train_dir + "Images/"
            label_img__dir_path = self.label_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split(".png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = label_img__dir_path + "TrainId/" + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_path = example["img_path"]
        label_img_path = example["label_img_path"]
        img = Image.open(img_path).convert('RGB')
        label_img = Image.open(label_img_path)

        if self.transform:
            img, label_img = self.transform(img, label_img)

        return (img, label_img)


    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, uavid_data_path, uavid_meta_path, transform=None):
        self.img_dir = uavid_data_path + "/valid/"
        self.label_dir = uavid_meta_path + "/labelimg/valid/"
        self.transform = transform
        self.img_h = 2160
        self.img_w = 3840

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for val_dir in val_dirs:
            val_img_dir_path = self.img_dir + val_dir + "Images/"
            label_img__dir_path = self.label_dir + val_dir

            file_names = os.listdir(val_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split(".png")[0]

                img_path = val_img_dir_path + file_name

                label_img_path = label_img__dir_path + "TrainId/" + img_id + ".png"
                # label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example["img_id"]
        img_path = example["img_path"]
        label_img_path = example["label_img_path"]
        img = Image.open(img_path).convert('RGB')
        label_img = Image.open((label_img_path))
        if self.transform:
            img, label_img = self.transform(img, label_img)

        return (img, label_img, img_id)

    def __len__(self):
        return self.num_examples



class DatasetSeq(torch.utils.data.Dataset):
    def __init__(self, uavid_data_path, uavid_meta_path, sequence):
        self.img_dir = uavid_data_path + "/demoVideo/stuttgart_" + sequence + "/"
        # self.img_dir = cityscapes_data_path + "/leftImg8bit/" + sequence + "/"

        self.img_h = 2160
        self.img_w = 3840

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split(".png")[0]

            img_path = self.img_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        print(img_path)
        img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
        print(img.shape)
        # resize img without interpolation:
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples

def decode_target(target):
    target[target == 255] = 19
    train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
                         (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    train_id_to_color = np.array(train_id_to_color)
    return train_id_to_color[target]
