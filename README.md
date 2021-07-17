# CCVAE
Final year masters project code.

To run the code use **Wrapper**, this file runs all the subfiles to complete the analysis. For this to work you must have a dataset of Nifti 3D MRI images, the filepath is hard-coded into the **MRI_CVAE.py** file and so this must be changed otherwise an error will show, the dataset for this analysis is not publically availabe.

The project has been split into distinct files which all serve individual purpose but interconnect with each other and so all files must be in the same directory for the code to work. 

A description of each file:

**Wrapper** - The file which runs the whole analysis, this runs the preprocessing first and then the model before outputting all plots and tables that are made during the analysis. For this some files will be saved into the directory and so a file can be made to store these ahead of time. 

**MRI_CVAE** - This is the conditional convolutional variational autoencoder model, all image preprocessing is done in this script and the model architecture can be seen. No further analysis is found in this file.

**Slice_variation_analysis** - This file is an analysis of input slice variation, to determine whether the use of ROI is worthwhile in this regard or whether slice variation is differential enough to use a more unbiased approach. Slice variation is differential but not enough to warrent using this method. It may be beneficial for further research. 

**CVAE_3D_plots** - All plotting functions for the analysis are here. 

**ccvae_analysis** - This is the further analysis of the model, LDA is included in this along with SSIM measurements and reconstruction variation analysis. 

Many other files were created throughout the course of this project, these are not included in this repo to make it easier to read. If interested the full file repo for this project are located at https://github.com/SophieWilson/VariationalConditionalAutoencoder
