import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm

# Load images as greyscale but make main RGB so we can annotate in colour
def visualization(seg_path, img_path, visual_path):

    if os.path.exists(visual_path) and os.path.isdir(visual_path):
        shutil.rmtree(visual_path)
    try:
        os.mkdir(visual_path)
    except OSError:
        print("Creation of the main directory '%s' failed " % visual_path)

    for image in os.listdir(seg_path):

        seg = seg_path + '/' + image
        main = img_path + '/' + image

        seg = cv2.imread(seg, cv2.IMREAD_GRAYSCALE)
        main = cv2.imread(main, cv2.IMREAD_GRAYSCALE)

        # seg = np.resize(seg, (main.shape[0], main.shape[1]))
        main = cv2.cvtColor(main,cv2.COLOR_GRAY2BGR)

        # Dictionary giving RGB colour for label (segment label) - label 1 in red, label 2 in yellow
        RGBforLabel = { 1:(255,0,255), 2:(0,255,255) }

        # Find external contours
        contours,_ = cv2.findContours(seg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # Iterate over all contours
        for i,c in enumerate(contours):

            # Find mean colour inside this contour by doing a masked mean
            mask = np.zeros(seg.shape, np.uint8)
            cv2.drawContours(mask,[c],-1,255, -1)

            # DEBUG: cv2.imwrite(f"mask-{i}.png",mask)
            mean,_,_,_ = cv2.mean(seg, mask=mask)

            # DEBUG: print(f"i: {i}, mean: {mean}")
            # Get appropriate colour for this label
            label = 2 if mean > 1.0 else 1

            colour = RGBforLabel.get(label)

            # DEBUG: print(f"Colour: {colour}")
            # Outline contour in that colour on main image, line thickness=1
            cv2.drawContours(main,[c],-1,colour,1)
            # Save result
        cv2.imwrite(visual_path + '/' + image, main)

visualization(r"./test_data/predict", # mask path
              r"./test_data/img",     # image path
              r"./test_data/visual")   # visualize path



