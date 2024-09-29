# Digital-rock-construction

![Image text](https://github.com/always258/Digital-rock-construction/blob/main/Digital%20rock%20construction/img_folder/1-1.png)


## Prerequisites

- Python 3.8
- PyTorch 1.12.0 + cu113 
- NVIDIA GPU + CUDA cuDNN

## Installation

- Clone this repo:

  ```
  git clone https://github.com/always258/rock.git
  ```

- Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org/)

- Install python requirements:

  ```
  pip install -r requirements.txt
  ```

## Started



###   Coarse scale 3D reconstruction

1、First, we need to remove the fine-scale structure from data that has both coarse-scale and fine-scale.


2、Run the code CRockGAN.py for training and generating coarse-scale 3D structures. The address is as follows "./Coarse-scale-3D-reconstruction/rockgan/CRockGAN.py"

```
python CRockGAN.py 
```



### Injecting fine-scale information in three dimensions

#### Trainting

To train the model on an anisotropic material with limited training data , run:

```
python code/Architecture.py -d rock --separator --anisotropic -phases_idx 1 -sf 4 -g_image_path lr75.tif -d_image_path hr32.tif hr32.tif hr6.tif
```


#### Testing

With the same directory name chosen for training. Specify for the size of the low-res volume to be super-resolved. There is no need to specify or here since only the generator is used. 
```
python code/Evaluation.py -d rock -phases_idx 1 -sf 4 -volume_size_to_evaluate 156 75 75 -g_image_path lr75.ti
```


## Example
Below is a display of some of our results, the original images and results can be viewed in the file ``` /img_folder/example/ ```.

![Image text](https://github.com/always258/Digital-rock-construction/blob/main/Digital%20rock%20construction/img_folder/11.png)

## Acknowledgements

We would like to thank [WGAN-GP](https://github.com/ChenKaiXuSan/WGAN-GP-PyTorch.git), and [SliceGAN](https://github.com/stke9/SliceGAN.git).

If we missed a contribution, please contact us.
