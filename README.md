# H3D-Net-reproduction
The goal of this blog post is to present and describe our implementation to reproduce the deep learning paper “H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction” using Pytorch. We are doing this for an assignment of the course CS4240 Deep Learning (2021/22 Q3) at Delft University of Technology. This paper introduces a new high-fidelity full 3D head reconstruction method called H3D-Net that outperforms state-of-the-art models, such as MVFNet, DFNRMVS and IDR, in the few-shot (3 views) scenario. The H3D-Net utilizes both DeepSDF (a learned shape prior) and IDR (fine-tuning details) to achieve fast high-fidelity 3D face reconstruction from 2D images with different views. Please check the papers for more background information about DeepSDF and IDR. Our approach attempts to reproduce the results of the last two rows of "Table 2: 3D Reconstruction Method Comparison" of the paper.

![Table 2 ](README_images/Table2.png)

1. Paper describing IDR method: https://lioryariv.github.io/idr/  
   - Repository for IDR method: https://github.com/lioryariv/idr
2. Paper describing DeepSDF method: https://paperswithcode.com/paper/deepsdf-learning-continuous-signed-distance  
   - Repository for DeepSDF method: 
3. Paper describing H3D-Net method: https://crisalixsa.github.io/h3d-net/  
   - Repository for H3D-Net method: Source code is not available, only the code used to manipulate the H3DS Dataset has been supplied: https://github.com/CrisalixSA/h3ds

## Reproducibility Approach

In order to reproduce the results from Table 2, we will first start by implementing the IDR method as described in Paper 1 using the H3DS dataset. We believe this will take a considereable amount of time to implement due to the large amount of training needed. Each scan_id (person's head) needs to be trained on for 3, 4, 8, 16, 32 views. The results from this implementation should give us the second last row of Table 2.
Once implemented we will move on to the DeepSDF method described in paper 2, followed by the actual H3D-Net implementation. 

One thing to note is that by the end of this reproducibility project we will have reimplemented methods described in 3 different papers, which is a condsidereable amount of work given the short amount of time allocated. (4-5 weeks).

## Google Cloud Platform

bla bla
 
## IDR Method

The IDR method source code has been supplied and can be found in Repository 1, however it is taylored to the DTU dataset. So a few modiciations were needed in order to use the H3DS Dataset. 

First we cloned the repository of Paper 1 into the folder called `IDR` (as can be seen in our repository above). You can do this by calling the following:

```
git clone https://github.com/lioryariv/idr
```

The we need to create the idr evironment. (These instructions can be found in the `README.md` file of the `IDR` repository, but we will list it here to get you going)

```
conda env create -f environment.yml
conda activate idr
```

Now everything should be setup. 

### IDR Training

In order to start training the H3DS Dataset, we had to apply for permission to use their dataset, in order to get the `H3DS_ACCESS_TOKEN`, and in order to use the H3DS dataset, we used the repository of Paper 3, in order to be able to manipulate the data.

We cloned repository 3 in the `idr-main` folder of this repository.

```
git clone https://github.com/CrisalixSA/h3ds
```

Now we have a means of working with the H3DS Dataset.

#### data_processing.py

`data_processing.py` uses the H3DS Dataset and organises the `images`, `masks` and `cameras.npz` files into different `views` (3, 4, 8, 16, 32). 

After running this python file, we are left with a folder called `OWN_DATA`, which contains folders with `view_ids`, which each contain all the `scan_id` folders, which each contain the respective `images`, `masks` and `cameras.npz` files for that scene. The data was split as shown in the `h3ds-main/h3ds/config.toml` file. This was done so that we could train different idr models when given different `views`.

#### H3D_fixed_cameras_X.conf

`H3D_fixed_cameras_X.conf` where `X` = `views` (3, 4, 8, 16, 32). These 5 files were added to `IDR/code/confs/` as the H3DS image resolution was different to the original `DTU Dataset` used in the idr paper 1. This file also redirects the training data to our `OWN_DATA`.

We are now ready to train on the different views. In order to train, we used the following code:

```
cd ./code
python training/exp_runner.py --conf ./confs/H3D_fixed_cameras_X.conf --scan_id SCAN_ID
```
Where `SCAN_ID` is the scan number shown within each `view_id`. The link between `scan_id` and `scene_id` can be found in `OWN_DATA/scan_list.txt`. 






