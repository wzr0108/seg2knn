import json
import argparse
import os
from PIL import Image

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def hex_to_rgb(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return r, g, b


parser = argparse.ArgumentParser()
parser.add_argument('--xy_index', default="result/xy_index.txt", help='result of nucleus_in_spot.py')
parser.add_argument('--CARD-result', default="data/our_Proportion_CARD.csv", help="CARD deconvolution result")
parser.add_argument('--HE-path', '-p', default="data/HAIMA.tif", help='Path of HE')
parser.add_argument('--save-dir', default="result", type=str)
args = parser.parse_args()

xy_index = pd.read_csv(args.xy_index, sep='\t', header=None, index_col=0)
our_Proportion_CARD = pd.read_csv(args.CARD_result, sep=',', index_col=0)
unique_ct = [
    "Astrocyte", "CA1", "CA3", "Cajalretzius", "Choroid", "Dentate", "EndothelialStalk",
    "EndothelialTip", "Entorhinal", "Ependymal", "Interneuron", "macrophage", "Mural", "Neurogenesis",
    "NeuronSlc17a", "Oligodendrocyte", "Polydendrocyte"
]
colors_all = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080",
    "#008000", "#800000", "#008080", "#808000", "#000080", "#808080", "#FF4500", "#DA70D6",
    "#7B68EE"]
color_per_ct = dict(zip(unique_ct, colors_all[:len(unique_ct)]))

HE = Image.open(args.HE_path)
w, h = HE.size
img = np.ones((h, w, 3), dtype=np.uint8) * 255

xy_cls = []
x_list = []
y_list = []
type_list = []
for bc in xy_index.index:
    xy_list = xy_index.loc[bc, 4]
    if xy_list == "[]":
        continue
    xy_list = eval(xy_list)

    num = len(xy_list)
    prob2num = []

    for col in our_Proportion_CARD.columns:
        tmp = our_Proportion_CARD.loc[bc, :]
        prob = our_Proportion_CARD.loc[bc, col]

        expect_num = prob * num

        if expect_num > int(expect_num):
            expect_num = int(expect_num) + 1

        prob2num.append([prob, expect_num, col])

    select_num = sum([each[1] for each in prob2num])

    prob2num = sorted(prob2num, key=lambda z: z[0])
    pointer = 0

    while select_num > len(xy_list):
        prob2num[pointer][1] -= 1
        pointer += 1
        select_num -= 1

    candidate = list(range(len(xy_list)))
    for prob, num, cls in prob2num:
        if num == 0:
            continue
        else:
            select = np.random.choice(candidate, num, replace=False).tolist()
            for s in select:
                xy = xy_list[s]
                x = xy[0]
                y = xy[1]
                x_list.append(x)
                y_list.append(y)
                type_list.append(cls)
                xy_cls.append([x, y, cls])
            candidate = list(set(candidate) - set(select))

    if select_num != len(xy_list):
        print(len(xy_list) - select_num)

x_list = np.array(x_list)
y_list = np.array(y_list)
type_list = np.array(type_list)
colors = np.array([color_per_ct[cls] for cls in type_list])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
for cls in np.unique(type_list):
    mask = type_list == cls
    x = x_list[mask]
    y = y_list[mask]
    plt.scatter(x, y, c=colors[mask], s=3, label=cls)

ax.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])

ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

plt.savefig(os.path.join(args.save_dir, "nucleus_type.pdf"), dpi=300, bbox_inches='tight')
# plt.show()

df = pd.DataFrame(
    {
        'x': x_list,
        'y': y_list,
        'type': type_list
    }
)
df.to_csv(os.path.join(args.save_dir, "nucleus_xy_type.txt"), index=False)
