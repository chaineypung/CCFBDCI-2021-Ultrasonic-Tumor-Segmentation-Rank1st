from __future__ import print_function, division
from dataset import Train_Dataset, Valid_Dataset
from torch.utils.data import DataLoader
import shutil
import argparse
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from metrics import *
from util import *
import segmentation_models_pytorch_4TorchLessThan120 as smp
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main(args):

    # load args
    input_channel = args.input_channel
    output_class = args.output_class
    image_resolution = args.image_resolution
    epoch = args.epochs
    num_workers = args.num_workers
    device = args.device
    batch_size = args.batch_size
    backbone = args.backbone
    network = args.network
    initial_lr = args.initial_learning_rate
    MAX_STEP = args.t_max
    K = args.folds
    fold = args.k_th_fold
    fold_file_list = args.fold_file_list
    train_dataset_path = args.train_dataset_path
    train_gt_dataset_path = args.train_gt_dataset_path
    New_folder = args.saved_model_path
    read_pred = args.visualize_of_data_aug_path
    weights_path = args.weights_path
    weights = args.weights

    # check GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('Training on CPU')
    else:
        print(f'Training on GPU {device}')
    cuda = "cuda:" + str(device)
    device = torch.device(cuda if train_on_gpu else "cpu")
    print('image_size = ' + str(image_resolution))
    print('batch_size = ' + str(batch_size))
    print('epoch = ' + str(epoch))

    # initial params
    valid_loss_min = np.Inf
    lossT, lossL = [], []
    lossL.append(np.inf)
    lossT.append(np.inf)
    epoch_valid = epoch - 2
    n_iter, i_valid, model_test = 1, 0, 0

    # set pin_memory
    pin_memory = False
    if train_on_gpu:
        pin_memory = True

    # select backbone and network
    if network == "Linknet":
        model_test = smp.Linknet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel, classes=output_class)
    if network == "DeepLabV3Plus":
        model_test = smp.DeepLabV3Plus(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel, classes=output_class)
    if network == "FPN":
        model_test = smp.FPN(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel, classes=output_class)
    if network == "PAN":
        model_test = smp.PAN(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel, classes=output_class)
    if network == "PSPNet":
        model_test = smp.PSPNet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel, classes=output_class)
    if network == "Unet":
        model_test = smp.Unet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel, classes=output_class)
    model_test.to(device)

    # split train set and valid set
    train, valid = get_fold_filelist(fold_file_list, K, fold)
    train_list = [train_dataset_path + sep + i[0] for i in train]
    train_list_GT = [train_gt_dataset_path + sep + i[0] for i in train]
    valid_list = [train_dataset_path + sep + i[0] for i in valid]
    valid_list_GT = [train_gt_dataset_path + sep + i[0] for i in valid]
    print(f"Dataset has been divided by calculating mask areas")
    print(f"{fold} / {K} fold training")

    # set DataLoader
    train_data = Train_Dataset(img_list=train_list, label_list=train_list_GT, image_resolution=image_resolution)
    valid_data = Valid_Dataset(img_list=valid_list, label_list=valid_list_GT, image_resolution=image_resolution)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_data, batch_size=10, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # set optimizer
    opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(MAX_STEP), eta_min=1e-11)

    # set checkpoint
    if os.path.exists(New_folder) and os.path.isdir(New_folder):
        shutil.rmtree(New_folder)
    try:
        os.mkdir(New_folder)
    except OSError:
        print("Creation of the main directory '%s' failed " % New_folder)
    # else:
    #     print("Successfully created the main directory '%s' " % New_folder)
    if os.path.exists(read_pred) and os.path.isdir(read_pred):
        shutil.rmtree(read_pred)
    try:
        os.mkdir(read_pred)
    except OSError:
        print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
    # else:
    #     print("Successfully created the prediction directory '%s' of dice loss" % read_pred)
    read_model_path = weights_path
    if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
        shutil.rmtree(read_model_path)
        print('Model folder there, so deleted for newer one')
    try:
        os.mkdir(read_model_path)
    except OSError:
        print("Creation of the model directory '%s' failed" % read_model_path)
    # else:
    #     print("Successfully created the model directory '%s' " % read_model_path)

    # start training
    for i in range(epoch):

        train_loss = 0.0
        valid_loss = 0.0
        scheduler.step(i)
        lr = scheduler.get_lr()

        model_test.train()
        k = 1
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            input_images(x, y, i, n_iter, k)
            opt.zero_grad()
            y_pred = model_test(x)
            lossT = calc_loss(y_pred, y)
            lossT.backward()
            opt.step()
            train_loss += lossT.item() * x.size(0)
            k = 2

        model_test.eval()
        with torch.no_grad():
            for x1, y1 in valid_loader:
                x1, y1 = x1.to(device), y1.to(device)
                y_pred = model_test(x1)
                # y_pred11 = F.sigmoid(y_pred1)
                lossL = calc_loss(y_pred, y1)
                valid_loss += lossL.item() * x1.size(0)
            train_loss = train_loss / len(train_list)
            valid_loss = valid_loss / len(valid_list)
            if (i + 1) % 1 == 0:
                print('Epoch: {}/{} Training Loss: {:.6f} Validation Loss: {:.6f} Learning Rate: {:.9f}'.format(i + 1, epoch, train_loss, valid_loss, lr[0]))
            if valid_loss <= valid_loss_min and epoch_valid >= i:
                print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model '.format(valid_loss_min, valid_loss))
                torch.save(model_test.state_dict(), weights)
                valid_loss_min = valid_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="UL SEG", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_channel', type=int, default=1, help='image channel')
    parser.add_argument('--output_class', type=int, default=1, help='output class, binary classification (output_class = 1)')
    parser.add_argument('--image_resolution', type=int, default=256, help='image resolution we resize')
    parser.add_argument('--epochs', type=int, default=100, help='max epoch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--backbone', type=str, default="resnet34", help='backbone')
    parser.add_argument('--network', type=str, default="Linknet", help='network')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-7, help='initial learning rate')
    parser.add_argument('--t_max', type=int, default=110, help='CosineAnnealingLR parameter')
    parser.add_argument('--folds', type=int, default=5, help='split number')
    parser.add_argument('--k_th_fold', type=int, default=1, help='k-th fold we train')

    parser.add_argument('--fold_file_list', type=str, default="./train_data/train.csv", help='fold file list')
    parser.add_argument('--train_dataset_path', type=str, default="./train_data/img", help='train dataset path')
    parser.add_argument('--train_gt_dataset_path', type=str, default="./train_data/label", help='train ground truth path')
    parser.add_argument('--saved_model_path', type=str, default="./saved_model", help='saved model path')
    parser.add_argument('--visualize_of_data_aug_path', type=str, default="./saved_model/pred", help='visualization data augmentation')
    parser.add_argument('--weights_path', type=str, default="./saved_model/weights", help='weights path')
    parser.add_argument('--weights', type=str, default="./saved_model/weights/best_model.pth", help='best_model.pth')

    args, unkown = parser.parse_known_args()
    main(args)


