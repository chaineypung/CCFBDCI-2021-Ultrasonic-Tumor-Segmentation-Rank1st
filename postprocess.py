from PIL import Image
import numpy as np
from skimage import measure
import cv2
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

def main(args):
    image_path = args.image_path
    threshood = args.threshood
    kernel = args.kernel
    remove_small_points(image_path, threshood)
    MedianFilter(image_path, kernel)


def remove_small_points(image_path, threshold_point):
    print("start remove small points...")
    for image in os.listdir(image_path):
        image_file_path = image_path + '/' + image
        # input binary image
        img_tmp = cv2.imread(image_file_path, 0)
        # output all connected domains in the binary image
        img_label, num = measure.label(img_tmp, neighbors=8, return_num=True)
        # properties of the output concatenation field
        props = measure.regionprops(img_label)
        resMatrix = np.zeros(img_label.shape)
        for i in range(1, len(props)):
            if props[i].area > threshold_point:
                tmp = (img_label == i + 1).astype(np.uint8)
                # combine all eligible connected domains
                resMatrix += tmp
        resMatrix *= 255
        resMatrix = img_tmp - resMatrix
        print(f"{image} done!")
        cv2.imwrite(image_file_path, resMatrix)


def MedianFilter(image_path, kernel):
    print("start median filter...")
    for image in os.listdir(image_path):
        srcc = image_path + '/' + image
        imarray = np.array(Image.open(srcc))
        height = imarray.shape[0]
        width = imarray.shape[1]
        # kernel size can be adjusted
        edge = int((kernel - 1) / 2)
        if height - 1 - edge <= edge or width - 1 - edge <= edge:
            print("The parameter k is to large.")
            return None
        new_arr = np.zeros((height, width), dtype="uint8")
        for i in range(height):
            for j in range(width):
                if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
                    if imarray.ndim == 2:
                        new_arr[i, j] = imarray[i, j]
                    else:
                        new_arr[i, j] = imarray[i, j, 0]
                else:
                    if imarray.ndim == 2:
                        new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])
                    else:
                        new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1, 0])
        new_im = Image.fromarray(new_arr)
        new_im.save(srcc)
        print(f"{image} done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="POSTPROCESS", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path', type=str, default="./test_data/img", help='image path')
    parser.add_argument('--threshood', type=int, default=50, help='small point threshold')
    parser.add_argument('--kernel', type=int, default=20, help='median filter kernel')

    args, unkown = parser.parse_known_args()
    main(args)

