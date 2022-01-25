import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from imgaug import augmenters as iaa
import imgaug as ia

class Train_Dataset(Dataset):

    def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None):

        self.img_dir = img_list
        self.label_dir = label_list

        self.transform = transform
        self.target_transform = target_transform

        self.image_resolution = image_resolution

    def __getitem__(self, idx):

        # normMean = [0.2218127, 0.22180918, 0.22183698]
        # normStd = [0.18524475, 0.18523921, 0.18525602]
        # normTransform = transforms.Normalize(normMean, normStd)

        # seed = np.random.randint(2147483647)
        # p1 = random.randint(0, 1)
        # p2 = random.randint(0, 1)

        image_name = self.img_dir[idx]
        label_name = self.label_dir[idx]

        img = Image.open(image_name).convert('RGB')
        label = Image.open(label_name).convert('RGB')

        img_shape = np.array(img)

        self.transform = transforms.Compose([
            # transforms.Resize(32),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(p1),
            # transforms.RandomVerticalFlip(p2),
            # transforms.RandomRotation(10, resample=False, expand=False, center=None),
            # transforms.RandomAffine(degrees = (0, 20), scale=(0.8, 1)),
            # transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            # transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # normTransform,
            transforms.Grayscale(num_output_channels=1),
        ])
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        img = self.transform(img)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 设定随机函数,50%几率扩增,or
        # sometimes = lambda aug: iaa.Sometimes(0.7, aug)     # 设定随机函数,70%几率扩增

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.3),  # 50%图像进行水平翻转
                iaa.Flipud(0.3),  # 50%图像做垂直翻转

                sometimes(iaa.Crop(percent=(0, 0.05))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%
                # sometimes(iaa.Crop(percent=(0, 0.2))),  # 对随机的一部分图像做crop操作 crop的幅度为0到20% wang

                sometimes(iaa.Affine(  # 对一部分图像做仿射变换
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
                    rotate=(-10, 10),  # 旋转±45度之间
                    shear=(-5, 5),  # 剪切变换±16度，（矩形变平行四边形）
                    order=[0, 1],  # 使用最邻近差值或者双线性差值
                    cval=(0, 255),
                    mode=ia.ALL,  # 边缘填充
                )),

                # 使用下面的0个到4个之间的方法去增强图像
                iaa.SomeOf((0, 5),
                           [
                               # 锐化处理
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)),

                               # 扭曲图像的局部区域
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),

                               # 改变对比度
                               iaa.contrast.LinearContrast((0.75, 1.25), per_channel=0.5),

                               # 用高斯模糊，均值模糊，中值模糊中的一种增强
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),
                                   iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),

                               # 加入高斯噪声
                               iaa.AdditiveGaussianNoise(
                                   loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                               ),

                               # 边缘检测，将检测到的赋值0或者255然后叠在原图上(不确定)
                               # sometimes(iaa.OneOf([
                               #     iaa.EdgeDetect(alpha=(0, 0.7)),
                               #     iaa.DirectedEdgeDetect(
                               #         alpha=(0, 0.7), direction=(0.0, 1.0)
                               #     ),
                               # ])),

                               # 浮雕效果(很奇怪的操作,不确定能不能用)
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                               # 将1%到10%的像素设置为黑色或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                   iaa.CoarseDropout(
                                       (0.03, 0.15), size_percent=(0.02, 0.05),
                                       per_channel=0.2
                                   ),
                               ]),

                               # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                               sometimes(
                                   iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                               ),

                           ],

                           random_order=True    # 随机的顺序把这些操作用在图像上
                           )
            ],
            random_order=True  # 随机的顺序把这些操作用在图像上

        )

        self.target_transform = transforms.Compose([
            # transforms.Resize(32),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(p1),
            # transforms.RandomVerticalFlip(p2),
            # transforms.RandomRotation(10, resample=False, expand=False, center=None),
            # transforms.RandomAffine(degrees=(0, 20), scale=(0.8, 1)),
            transforms.Resize((img_shape.shape[0], img_shape.shape[1]), interpolation=Image.NEAREST),
            transforms.Grayscale(num_output_channels=1),
            # transforms.ToTensor(),
        ])
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        label = self.target_transform(label)

        img = np.array(img)
        label = np.array(label).astype(np.uint8)

        seq_det = seq.to_deterministic()  # 确定一个数据增强的序列
        # print('imgs.shape',imgs.shape)
        images_aug = seq_det.augment_images(img)  # 进行增强
        segmaps = ia.SegmentationMapsOnImage(label, shape=label.shape)  # 分割标签格式
        segmaps_aug = seq_det.augment_segmentation_maps(segmaps).get_arr().astype(np.uint8)  # 将方法应用在分割标签上，并且转换成np类型

        images_aug = Image.fromarray(images_aug)
        segmaps_aug = Image.fromarray(segmaps_aug)

        Transform = []
        Transform_GT = []

        # Transform.append(transforms.Resize((256, 256), interpolation=Image.BICUBIC))
        # Transform_GT.append(transforms.Resize((256, 256), interpolation=Image.NEAREST))

        Transform.append(transforms.Resize((self.image_resolution, self.image_resolution), interpolation=Image.BICUBIC))
        Transform_GT.append(transforms.Resize((self.image_resolution, self.image_resolution), interpolation=Image.NEAREST))

        Transform.append(transforms.ToTensor())
        Transform_GT.append(transforms.ToTensor())

        Transform = transforms.Compose(Transform)
        Transform_GT = transforms.Compose(Transform_GT)

        images_aug = Transform(images_aug)
        segmaps_aug = Transform_GT(segmaps_aug)

        return images_aug, segmaps_aug

    def __len__(self):
        return len(self.img_dir)

class Valid_Dataset(Dataset):

    def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None):

        self.img_dir = img_list
        self.label_dir = label_list
        self.transform = transform
        self.target_transform = target_transform
        self.image_resolution = image_resolution

    def __getitem__(self, idx):

        image_name = self.img_dir[idx]
        label_name = self.label_dir[idx]

        img = Image.open(image_name).convert('RGB')
        label = Image.open(label_name).convert('RGB')

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_resolution, self.image_resolution)),
            transforms.ToTensor(),
        ])
        img = self.transform(img)

        self.target_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_resolution, self.image_resolution)),
            transforms.ToTensor(),
        ])
        label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.img_dir)

class Test_Dataset(Dataset):

    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.img_path = os.listdir(self.img_dir)
        self.transform = transform

    def __getitem__(self, idx):

        image_name = self.img_path[idx]
        img_item_path = os.path.join(self.img_dir, image_name)
        img = Image.open(img_item_path).convert('RGB')
        img_array = np.array(img)
        w = img_array.shape[0]
        h = img_array.shape[1]
        if self.transform is not None:
            img = self.transform(img)
        return img, w, h, image_name

    def __len__(self):
        return len(self.img_path)