import os
import pandas as pd
from PIL import Image
import argparse
import json

import numpy as np
import cv2
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--HE-path', '-p', default="data/HAIMA.tif", help='Path of HE')
parser.add_argument('--json-file', '-j', default="result/HAIMA.json", help='json file of prediction')
parser.add_argument('--spot-example', '-s', default="data/20230315162704.png", type=str, help='example image of spot')
parser.add_argument('--save-dir', default="result", type=str)
parser.add_argument('--xy_index', default="data/xy_index_y6.txt", type=str, help='barcode and spot')
args = parser.parse_args()

HE = np.array(Image.open(args.HE_path))
pred_json = json.load(open(args.json_file, 'r'))
nuclei_centers = [each["centroid"] for each in pred_json['nuc'].values()]
nuclei_contours = [np.array(each["contour"]) for each in pred_json['nuc'].values()]
pred_instance = np.zeros(HE.shape[:2], dtype=np.int32)

for i, contour in enumerate(nuclei_contours):
    cv2.drawContours(pred_instance, [contour], 0, i + 1, -1)  # pixel i stands for the i-th nuclei

spot_img = cv2.imread(args.spot_example)
spot_img_gray = cv2.cvtColor(spot_img, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------------------------------------------------#
# Find the coordinates of the spot, and finally use k means clustering to get 50 x-coord and y-coord
circle1 = cv2.HoughCircles(spot_img_gray, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=30, minRadius=10,
                           maxRadius=20)
circles = circle1[0, :, :]  # 提取为二维
circles = np.uint16(np.around(circles))
spot_xlist = []
spot_ylist = []
for x, y, r in circles:
    spot_xlist.append(x)
    spot_ylist.append(y)

spot_xlist = np.array(spot_xlist)
spot_ylist = np.array(spot_ylist)

km = KMeans(n_clusters=50, max_iter=1000).fit(spot_xlist.reshape(-1, 1))
spot_x_cluster = km.cluster_centers_.copy().reshape(-1)

km2 = KMeans(n_clusters=50, max_iter=1000).fit(spot_ylist.reshape(-1, 1))
spot_y_cluster = km2.cluster_centers_.copy().reshape(-1)

spot_r = np.median(circles[:, 2])

draw_spot = spot_img.copy()
for y in spot_y_cluster:
    for x in spot_x_cluster:
        x = int(x)
        y = int(y)
        cv2.circle(draw_spot, (x, y), int(spot_r), (0, 200, 0), 2)
        cv2.circle(draw_spot, (x, y), 2, (0, 0, 200), -1)
# -------------------------------------------------------------------------------------------------------------#


"""
spot_y_cluster: The y absolute coordinate of spot
spot_x_cluster: The x absolute coordinate of spot
spot_r:         radius of spot
"""
HE_draw = HE.copy()
spot_r = int(spot_r / spot_img_gray.shape[1] * HE_draw.shape[1])
spot_y_cluster = np.sort(spot_y_cluster)[::-1] / spot_img_gray.shape[0] * HE_draw.shape[0]
spot_x_cluster = np.sort(spot_x_cluster) / spot_img_gray.shape[1] * HE_draw.shape[1]

for i, contour in enumerate(nuclei_contours):
    cv2.drawContours(HE_draw, [contour], 0, (0, 0, 200), 2)

nuclei_counts = [0] * 2500
nuclei_spots = [[] for _ in range(2500)]
ptr = 0
for x in spot_x_cluster:
    for y in spot_y_cluster:
        x0 = x - spot_r
        x1 = x + spot_r
        y0 = y - spot_r
        y1 = y + spot_r

        slc = pred_instance[int(y0):int(y1) + 1, int(x0):int(x1) + 1]  # pred_instance的i对应的是nuclei_centers[i-1]
        nuclei_in = np.unique(slc)
        if np.any(nuclei_in == 0):
            nuclei_in = nuclei_in[1:]  # remove 0
        nuclei_counts[ptr] += len(nuclei_in)

        for ni in nuclei_in:
            cx, cy = int(nuclei_centers[ni - 1][0]), int(nuclei_centers[ni - 1][1])
            # coordinate transformation
            cy = HE.shape[0] - cy - 1
            nuclei_spots[ptr].append((cx, cy))

        ptr += 1

        x = int(x)
        y = int(y)
        cv2.rectangle(HE_draw, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 200), 2)
        # cv2.circle(HE_draw, (x, y), spot_r, (0, 0, 200), 2)

Image.fromarray(HE_draw).save(os.path.join(args.save_dir, os.path.basename(args.HE_path)))

xy_index = pd.read_csv(args.xy_index, sep='\t', header=None)
xy_index.insert(3, '3', nuclei_counts)
xy_index.insert(4, '4', nuclei_spots)
xy_index.to_csv(os.path.join(args.save_dir, "xy_index.txt"), sep='\t', header=None, index=None)
