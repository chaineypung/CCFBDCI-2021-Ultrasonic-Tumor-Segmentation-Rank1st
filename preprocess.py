import csv
import os
import cv2
import random
import argparse


def main(args):

    image_path = args.image_path
    csv_path = args.csv_path
    preprocess(image_path, csv_path)


def preprocess(image_path, csv_path):
    print("start preprocess...")
    f = open(csv_path, 'w', encoding='utf-8', newline = '')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["ID", "CATE", "size"])
    for image in os.listdir(image_path):
        image_file_path = image_path + '/' + image
        img_tmp = cv2.imread(image_file_path, 0)
        size = (img_tmp == 255).sum()
        p = random.randint(0, 1)
        csv_writer.writerow([image, p, size])
        print(f"{image} done!")
    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PREPROCESS", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path', type=str, default="./train_data/label", help='image path')
    parser.add_argument('--csv_path', type=str, default="./train_data/train.csv", help='csv path')

    args, unkown = parser.parse_known_args()
    main(args)
