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