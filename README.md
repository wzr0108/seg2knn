## Installation

### **Step 1.** Create a conda environment and activate it.

```
conda create --name py3.7 python=3.7
conda activate py3.7
```

### **Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).  If you are experienced with PyTorch and have already installed it, just skip this part.

e.g.

On GPU platforms:

```
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

On CPU platforms:

```
pip install torch==1.7.0+cpu torchvision==0.8.0+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### **Step 3.** Install mmcv

```
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```

### **Step 4.** Install python packages

```
pip install -r requirements.txt
```



## Nuclei Segmentation

Download the checkpoint from https://drive.google.com/file/d/1GrH1IgF-o0VfJUeSgq1b0KesDs3ZzStl/view?usp=sharing

and put it in ./checkpoints/

run the code

```
python test.py config.py checkpoints/iter_80000_unwrap.pth --format-only --eval-options imgfile_prefix=./result
```

the results will be saved in ./result/

```
result/HAIMA.json # segmentation results
result/HAIMA.npy  # instance map 
result/HAIMA.tif  # overlay of segmentation contour and H&E image
```

## Count Nuclei in Spot

```
python nucleus_in_spot.py -p data/HAIMA.tif, -j result/HAIMA.json -s data/20230315162704.png --save_dir result --xy_index data/xy_index_y6.txt

"""
-p: the path to the H&E image
-j: the path to the json file, segmentation results of the previous step
-s: an example image of spot, used to find the coordinates of the center of each spot
--save_dir: save dirctory
--xy_index: a file with three colunms, the first column is barcode, and the next two                 columns are xy coordinates
"""
```

the results will be saved in save_dir( default ./result/)

```
result/xy_index.txt  # barcode and corresponding number and location of nuclei
                     # for each line :
                     # barcode, col, row, number of nucleus, location of nucleus(x, y)
```

## Infer Cell Type (combined with deconvolution results)

```
python assign_cell_type.py --xy_index result/xy_index.txt --CARD-result data/our_Proportion_CARD.csv -p data/HAIMA.tif --save_dir result

"""
--xy_index: the result of the previous step
--CARD-result: the result of CARD deconvolution
-p: the path to the H&E image
--save_dir: save dirctory
"""
```

the results will be saved in save_dir( default ./result/)

```
result/nucleus_xy_type.txt  # location of nucleus and its type, the first two columns are                             #  the xy coordinates of the H&E image, the last column is 
                            # the cell type
result/nucleus_type.pdf  # visualization
```

## Infer Cell Types Outside The Spot

```
python knn.py --j result/HAIMA.json --xy_type result/nucleus_xy_type.txt -p data/HAIMA.tif --save_dir result

"""
-j: the path to the json file, segmentation results of the first step
--xy_type: the result of the previous step
-p: the path to the H&E image
--save_dir: save dirctory
"""
```

python knn.py

```
result/all_nucleus_xy_type.txt  # location of nucleus and its type
result/all_nucleus_type.pdf  # visualization
```

