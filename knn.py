import json
from tqdm import tqdm
import os
import argparse
from PIL import Image

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def hex_to_rgb(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return r, g, b


parser = argparse.ArgumentParser()
parser.add_argument('--json-file', '-j', default="result/HAIMA.json", help='json file of prediction')
parser.add_argument('--xy_type', default="result/nucleus_xy_type.txt",
                    help='xy and type of nucleus, result of assign_cell_type.py')
parser.add_argument('--HE-path', '-p', default="data/HAIMA.tif", help='Path of HE')
parser.add_argument('--save-dir', default="result", type=str)
args = parser.parse_args()

pred_json = json.load(open(args.json_file, 'r'))
xy_type = pd.read_csv(args.xy_type, sep=',', header=0, index_col=None)

HE = Image.open(args.HE_path)
w, h = HE.size
nuclei_centers = [each["centroid"] for each in pred_json['nuc'].values()]
# coordinate transformation
nuclei_centers = [[int(each[0]), h - int(each[1]) - 1] for each in nuclei_centers]

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

knn = KNeighborsClassifier()

train_data = np.array(xy_type.iloc[:, :2], dtype=np.int32)
train_y = np.array(xy_type.iloc[:, -1])

all_data = np.array(nuclei_centers, dtype=np.int32)

all_date_str = all_data.tolist()
train_data_str = train_data.tolist()
all_date_str = [f"{each[0]}_{each[1]}" for each in all_date_str]
train_data_str = [f"{each[0]}_{each[1]}" for each in train_data_str]

test_data_str = list(set(all_date_str) - set(train_data_str))

test_data = [list(map(int, each.split('_'))) for each in test_data_str]
test_data = np.array(test_data, dtype=np.int32)

knn.fit(train_data, train_y)

test_pred = knn.predict(test_data)

# concat and pyplot
all_xy = np.concatenate([train_data, test_data], axis=0)
classes = np.concatenate([train_y, test_pred], axis=0)
colors = np.array([color_per_ct[cls] for cls in classes])

df = pd.DataFrame(
    {
        'x': all_xy[:, 0],
        'y': all_xy[:, 1],
        'type': classes
    }
)
df.to_csv(os.path.join(args.save_dir, "all_nucleus_xy_type.txt"), index=False)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

for cls in np.unique(classes):
    mask = classes == cls
    cls_xy = all_xy[mask]
    plt.scatter(cls_xy[:, 0], cls_xy[:, 1], c=colors[mask], s=3, label=cls)

ax.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])

ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

plt.savefig(os.path.join(args.save_dir, "all_nucleus_type.pdf"), dpi=300, bbox_inches='tight')
