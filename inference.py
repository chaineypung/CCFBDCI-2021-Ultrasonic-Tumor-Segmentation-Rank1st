from __future__ import print_function, division
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.nn.functional as F
import torch.nn
import torchvision
from dataset import Test_Dataset
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from skimage.io import imread, imsave
import segmentation_models_pytorch_4TorchLessThan120 as smp
import ttach as tta
import warnings
import argparse
import shutil
warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

def main(args):

    # load args
    input_channel = args.input_channel
    output_class = args.output_class
    image_resolution = args.image_resolution
    device = args.device
    backbone = args.backbone
    network = args.network
    weights1 = args.weights1
    weights2 = args.weights2
    weights3 = args.weights3
    weights4 = args.weights4
    weights5 = args.weights5
    test_path = args.test_path
    saved_path = args.saved_path

    # check GPU
    cuda = "cuda:" + str(device)
    device = torch.device(cuda if train_on_gpu else "cpu")

    # select backbone and network
    model_test1 = 0
    if network == "Linknet":
        model_test1 = smp.Linknet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                                 classes=output_class).to(device)
    if network == "DeepLabV3Plus":
        model_test1 = smp.DeepLabV3Plus(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                                       classes=output_class).to(device)
    if network == "FPN":
        model_test1 = smp.FPN(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                             classes=output_class).to(device)
    if network == "PAN":
        model_test1 = smp.PAN(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                             classes=output_class).to(device)
    if network == "PSPNet":
        model_test1 = smp.PSPNet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                                classes=output_class).to(device)
    if network == "Unet":
        model_test1 = smp.Unet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                              classes=output_class).to(device)

    # load weights
    model_test2 = model_test1
    model_test3 = model_test1
    model_test4 = model_test1
    model_test5 = model_test1
    model_test1.load_state_dict(torch.load(weights1))
    model_test2.load_state_dict(torch.load(weights2))
    model_test3.load_state_dict(torch.load(weights3))
    model_test4.load_state_dict(torch.load(weights4))
    model_test5.load_state_dict(torch.load(weights5))

    # set TTA
    tta_trans1 = tta.Compose([
        tta.VerticalFlip(),
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
    ])
    tta_trans2 = tta.Compose([
        tta.VerticalFlip(),
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
    ])
    tta_trans3 = tta.Compose([
        tta.VerticalFlip(),
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
    ])
    tta_trans4 = tta.Compose([
        tta.VerticalFlip(),
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
    ])
    tta_trans5 = tta.Compose([
        tta.VerticalFlip(),
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
    ])
    model_test1 = tta.SegmentationTTAWrapper(model_test1, tta_trans1, merge_mode='mean')
    model_test2 = tta.SegmentationTTAWrapper(model_test2, tta_trans2, merge_mode='mean')
    model_test3 = tta.SegmentationTTAWrapper(model_test3, tta_trans3, merge_mode='mean')
    model_test4 = tta.SegmentationTTAWrapper(model_test4, tta_trans4, merge_mode='mean')
    model_test5 = tta.SegmentationTTAWrapper(model_test5, tta_trans5, merge_mode='mean')

    # freeze weights
    model_test1.eval()
    model_test2.eval()
    model_test3.eval()
    model_test4.eval()
    model_test5.eval()

    # set dataset
    trainTransform_img = transforms.Compose([
        torchvision.transforms.Resize((image_resolution, image_resolution)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    train_data = Test_Dataset(img_dir=test_path, transform=trainTransform_img)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)

    # create file
    if os.path.exists(saved_path) and os.path.isdir(saved_path):
        shutil.rmtree(saved_path)
    try:
        os.mkdir(saved_path)
    except OSError:
        print("Creation of the main directory '%s' failed " % saved_path)

    #start inference
    print("start inference...")
    with torch.no_grad():
        for x, w, h, name in train_loader:
            x = x.to(device)
            name = str(name)
            name = only_numerics(name)

            y_pred1 = model_test1(x)
            y_pred1 = F.sigmoid(y_pred1)
            y_pred1 = torch.squeeze(y_pred1, dim=0)
            transform = Resize((w, h))
            y_pred1 = transform(y_pred1)
            y_pred1 = y_pred1.cpu().detach().numpy()

            y_pred2 = model_test2(x)
            y_pred2 = F.sigmoid(y_pred2)
            y_pred2 = torch.squeeze(y_pred2, dim=0)
            transform = Resize((w, h))
            y_pred2 = transform(y_pred2)
            y_pred2 = y_pred2.cpu().detach().numpy()

            y_pred3 = model_test3(x)
            y_pred3 = F.sigmoid(y_pred3)
            y_pred3 = torch.squeeze(y_pred3, dim=0)
            transform = Resize((w, h))
            y_pred3 = transform(y_pred3)
            y_pred3 = y_pred3.cpu().detach().numpy()

            y_pred4 = model_test4(x)
            y_pred4 = F.sigmoid(y_pred4)
            y_pred4 = torch.squeeze(y_pred4, dim=0)
            transform = Resize((w, h))
            y_pred4 = transform(y_pred4)
            y_pred4 = y_pred4.cpu().detach().numpy()

            y_pred5 = model_test5(x)
            y_pred5 = F.sigmoid(y_pred5)
            y_pred5 = torch.squeeze(y_pred5, dim=0)
            transform = Resize((w, h))
            y_pred5 = transform(y_pred5)
            y_pred5 = y_pred5.cpu().detach().numpy()

            y_pred6 = np.concatenate((y_pred1, y_pred2, y_pred3, y_pred4, y_pred5),axis=0)
            y_pred6 = np.mean(y_pred6, axis=0)

            y_pred7 = np.zeros((w, h))
            for i in range(0, y_pred6.shape[0], 1):
                for j in range(0, y_pred6.shape[1], 1):
                    if y_pred6[i, j] >= 0.5:
                        y_pred7[i, j] = 255
                    else:
                        y_pred7[i, j] = 0
            y_pred7 = y_pred7.astype('uint8')

            print(f"{name} done!")
            imsave(saved_path + '\{}.png'.format(name), y_pred7)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="UL PRE", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_channel', type=int, default=1, help='image channel')
    parser.add_argument('--output_class', type=int, default=1, help='output class, binary classification (output_class = 1)')
    parser.add_argument('--image_resolution', type=int, default=256, help='image resolution we resize')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--backbone', type=str, default="resnet34", help='backbone')
    parser.add_argument('--network', type=str, default="Linknet", help='network')

    parser.add_argument('--weights1', type=str, default="./saved_model/weights/best_model1.pth", help='best_model1.pth')
    parser.add_argument('--weights2', type=str, default="./saved_model/weights/best_model2.pth", help='best_model2.pth')
    parser.add_argument('--weights3', type=str, default="./saved_model/weights/best_model3.pth", help='best_model3.pth')
    parser.add_argument('--weights4', type=str, default="./saved_model/weights/best_model4.pth", help='best_model4.pth')
    parser.add_argument('--weights5', type=str, default="./saved_model/weights/best_model5.pth", help='best_model5.pth')

    parser.add_argument('--test_path', type=str, default="./test_data/img", help='test dataset path')
    parser.add_argument('--saved_path', type=str, default="./test_data/predict", help='prediction segmentation')

    args, unkown = parser.parse_known_args()
    main(args)