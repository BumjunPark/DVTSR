# DVTSR

PyTorch Implementation of "[Robust Temporal Super-Resolution for Dynamic Motion Video]()"

# Related work
PyTorch Implementation of "[Densely Connected Hierarchical Network for Image Denoising](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Park_Densely_Connected_Hierarchical_Network_for_Image_Denoising_CVPRW_2019_paper.pdf)", 
second place winner of sRGB track and third place winner of Raw-RGB track on [NTIRE 2019 Challenge on Real Image Denoising](http://www.vision.ee.ethz.ch/ntire19/). ([code](https://github.com/BumjunPark/DHDN))    

PyTorch reimplementation of "[Optical Flow Estimation using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850)" by Simon Niklaus. ([code](https://github.com/sniklaus/pytorch-spynet))

# Dependencies
Python 3.6    
PyTorch 1.0.0    
MATLAB    
TensorFlow    
TensorBoard    
TensorBoardX    
TorchSummary

# Data
We used [REDS](https://seungjunnah.github.io/Datasets/reds.html) VTSR dataset for training.    
To generate training data, use the matlab code generate_train.m

# Training
As an example, use the following command to use our training codes
```
python main_tsr.py --lr 1e-4 --step 2 --cuda True --train_data0 ./train_data0.h5 --train_data1 ./train_data1.h5 --train_label ./train_label.h5 --valid_data0 ./valid_data0.h5 --valid_data1 ./valid_data1.h5 --valid_label ./valid_label.h5 --gpu 0,1 --checkpoint ./checkpoint
```
If you want to train t=1/4, change 9th line of the main_tsr.py to from model_tsr_14, import model_tsr    
If you want to train t=1/2, change 9th line of the main_tsr.py to from model_tsr_12, import model_tsr    
If you want to train t=3/4, change 9th line of the main_tsr.py to from model_tsr_34, import model_tsr    
Also, you have to give right label for the target time.    
There are other options you can choose.    
Please refer to the code.  

# Test
As an example, use the following command to use our test codes
```
python test_tsr.py --cuda True --model0 ./mode0.pth --data ./path/to/data --gpu 0 --result0 ./result0/
```
There are other options you can choose.    
Please refer to the code.

# Contact
If you have any question about the code or paper, please feel free to contact kkbbbj@gmail.com

