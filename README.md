# Jupyter Notebook to Run Several Deep Learning-based Depth Estimation Algorithms

This repo contains jupyter notebooks to run a number of depth estimation 
algorithm and to visualize the result more easily. The models used in this 
repo includes:
* 2 monocular depth estimation models
    * [Monodepth2](https://github.com/nianticlabs/monodepth2)
    * [P<sup>2</sup>Net](https://github.com/svip-lab/Indoor-SfMLearner)
* 4 stereo depth estimation models
    * [PSMNet](https://github.com/JiaRenChang/PSMNet)
    * [GA-Net](https://github.com/feihuzhang/GANet)
    * [StereoNet](https://github.com/meteorshowers/StereoNet-ActiveStereoNet):
      *not the official implementation but very close to the model in the paper*
    * [AANet](https://github.com/haofeixu/aanet)

Please refer to the corresponding repos and papers for more details about 
the models.

### Test Images

The test images and their predicted results are in the `/imgs` directory. 
The structure for this folder is:
```
/imgs
    ├── /gt_depth
        ├── img1.csv
        ├── img2.csv
        ├── img3.csv
        ├── ...
    ├── /pred_depth
        ├── /monodepth2
            ├── img1.npy
            ├── img2.npy
            ├── img3.npy
            ├── ...
        ├── /p2net
            ├── img1.npy
            ├── img2.npy
            ├── img3.npy
            ├── ...
        ├── /stereonet
            ├── img1.npy
            ├── img2.npy
            ├── img3.npy
            ├── ...
        ├── ...
    ├── /left
        ├── img1.png
        ├── img2.png
        ├── img3.png
        ├── ...
    ├── /right
        ├── img1.png
        ├── img2.png
        ├── img3.png
        ├── ...
```

The `/left` and `/right` folders hold the rectified left and right images, respectively, 
as inputs in the format of `.png`. The `/gt_depth` folder stores the pseudo 
ground truth depth data obtained by the Intel D435 camera in the format of 
`.csv`. The `/pred_disp` directory contains 6 subfolders. Each subdolder is 
associated with one of the models and includes the predicted depth/disparity 
by that model in the form of numpy array `.npy`. Files with the same file indicate 
that they are the sensor outputs/prediction for the same time step. 

### Running the Notebook
Prior to running the any notebooks, install Anaconda and create an environment
by using `requirements.txt` in this repo. Note that some of the packages in 
the notebooks are not necessary. 

After creating the environment and installing the packages, activate the 
environment. Then you should be able to run each notebook using Jupyter Notebook.

### Result Comparison
TODO