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
python nucleus_in_spot.py
```

the results will be saved in ./result/

```
result/xy_index.txt  # barcode and corresponding number and location of nuclei
                     # for each line :
                     # barcode, col, row, number of nucleus, location of nucleus(x, y)
```

## Infer Cell Type (combined with deconvolution results)

```
python assign_cell_type.py
```

the results will be saved in ./result/

```
result/nucleus_xy_type.txt  # location of nucleus and its type
result/nucleus_type.pdf  # visualization
```

## Infer Cell Types Outside The Spot

```
python knn.py
```

python knn.py

```
result/all_nucleus_xy_type.txt  # location of nucleus and its type
result/all_nucleus_type.pdf  # visualization
```

