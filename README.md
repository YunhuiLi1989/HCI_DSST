# HCI_DSST
Code for "Densely Aggregated U-net with Spatial-Spectral Interaction Transformer for Hyperspectral Compressed Imaging Reconstruction"

The manuscript has been submitted to The Visual Computer, Springer Nature.

# Requirements

einops==0.7.0  
fvcore==0.1.5.post20221221  
imgvision==0.1.7.1  
numpy==1.25.2  
scipy==1.11.1  
thop==0.1.1.post2209072238  
torch==2.0.1  
torchsummary==1.5.1  
torchvision==0.15.2  

# Prepare Dataset
	We applied the same dataset as the following literature in our code: 
	      "Yuanhao Cai, Jing Lin, etal. Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction. In CVPR 2022".
	The download link for the dataset shared in the above literature is as follows: 
	training dataset: 'https://pan.baidu.com/share/init?surl=X_uXxgyO-mslnCTn4ioyNQ' (code: fo0q)
	testing dataset:  'https://pan.baidu.com/share/init?surl=LI9tMaSprtxT8PiAG1oETA' (code: efu8)
	
	Download the training dataset from 'https://pan.baidu.com/share/init?surl=X_uXxgyO-mslnCTn4ioyNQ' (code: fo0q) and place them to 'DSST\dataset\train_data'.
	Download the testing dataset from 'https://pan.baidu.com/share/init?surl=LI9tMaSprtxT8PiAG1oETA' (code: efu8) and place them to 'DSST\dataset\valid_data' and 'DSST\dataset\test_data'.


# Training

	Run "train_dsst_t.py".
	Run "train_dsst_s.py".
	Run "train_dsst_m.py".
	Run "train_dsst_l.py".


# Testing

	Download the pre-trained model from 'https://pan.baidu.com/s/1C1anWYqkOXViRswfbYpuzA' (code: DSST) and place them to 'DSST\model'.

	Run "test_dsst_t.py".
	Run "test_dsst_s.py".
	Run "test_dsst_m.py".
	Run "test_dsst_l.py".
