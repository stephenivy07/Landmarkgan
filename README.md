# Landmarkgan: Synthesizing faces from landmarks
This repository is the implementation of our work, [Landmarkgan: Synthesizing faces from landmarks](https://www.sciencedirect.com/science/article/pii/S016786552200215X)


## Preparation
```shell
1.landmark coordinates order:
    facial boundary landmarks are in the last.
    
2.pip install:
    torch(pytorch 1.0.1. Other versions might also work.)
    easydict
    dsntnn
    pyyaml
    opencv-python
    numpy
    pickle

Use opencv-python 3.x and ignore opencv version warning.

```

## Generation


### Step 1
* download lm_AE checkpoints from [here](https://drive.google.com/file/d/1xMNIxE5gotHS_30tpOdQ3t2qmQOiVe0e/view?usp=sharing).
* unzip it and put all files in ./model/ckpt/MODEL_0301_1_patchgan_lmconsis_lm_AE/
* Modify the absolute path in 00_FSNet_lm_AE.yml(line 20).

### Step 2
* download tl2f checkpoints from [here](https://drive.google.com/file/d/1Pvv7VvqTP3XnZSJ7JHX-YMr_9eTDO0Fo/view?usp=sharing).
* unzip it and put all files in ./model/ckpt/MODEL_0130_1_patchgan_lmconsis_one_decoder/
* Modify the absolute path in 00_FSNet_onedecoder.yml(line 20).

### Step 3
* python 0830_tkdemo.py
* [demo](https://drive.google.com/file/d/1ILUi4QVyvtsmBJumXvYHLdnSdZqosl1-/view?usp=sharing)


## Bibtex
```shell
@article{sun2022landmarkgan,
  title={Landmarkgan: Synthesizing faces from landmarks},
  author={Sun, Pu and Li, Yuezun and Qi, Honggang and Lyu, Siwei},
  journal={Pattern Recognition Letters},
  volume={161},
  pages={90--98},
  year={2022},
  publisher={Elsevier}
}
```