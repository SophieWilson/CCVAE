# This file should run the whole analysis
import numpy as np
from keras.utils import to_categorical
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
import nibabel as nib #reading MR images
import tensorflow as tf
from tensorflow import keras

import MRI_CVAE as cvae

# input data analysis

# Analysing data variation [takes ages so not running] [run for wholebrain only]
from slice_variation_analysis import slice_analysis, latent_ssim_analysis
#slice_analysis(cvae.images) # returns a dataframe

# plotting the input [WORKING]
from CVAE_3Dplots import plot_slices, sliceview, process_mwheel
plot_slices(cvae.x_test, n = 15) # plot
sliceview(cvae.x_test[1]) # shows mousewheel plot [WORKING]

# # # Model output reconstructions [WORKING]
from CVAE_3Dplots import reconstruction_plot, lossplot
reconstruction_plot_old(cvae.x_test, cvae.y_test, cvae.cvae, slice=2) # plot
reconstruction_plot(cvae.x_test, cvae.y_test, cvae.cvae) # superior plot
#lossplot(cvae.history) # plot [not working but not erroring
prediction = cvae.cvae.predict([cvae.x_test, cvae.y_test]) # [WORKING]
sliceview(prediction[0])

# Comparing variation between input and prediction data
from ccvae_analysis import structural_sim_data, var_boxplot, variation_summary
# var_boxplot calls structural_sim_data
var_boxplot(cvae.x_test, cvae.y_test, cvae.cvae)
# var_summary calls var_boxplot
variation_summary(cvae.x_test)

# Looking at latent space with LDA
from ccvae_analysis import lda
#lda(cvae.encoder, cvae.x_train, cvae.y_train, cvae.train_label)
scalings = lda(cvae.encoder, cvae.x_test, cvae.y_test, cvae.test_label)
scalings_df = pd.DataFrame(scalings)
df_scalings.to_csv(r'C:\Users\Mischa\Documents\Uni Masters\Diss project\Practise\MRI_cvae\df_scalings.csv')
# Plotting reconstruction latent space clusters over 2D
#from CVAE_3Dplots import plot_clusters
#plot_clusters(cvae.encoder, cvae.x_test, cvae.y_test, cvae.test_label, batch_size = 16)
#plot_clusters(cvae.encoder, cvae.x_train, cvae.y_train, cvae.train_label, batch_size = 16)

# Analysing lat space variance
lats_df, diff = latent_ssim_analysis(0, 10, 4, decoder, 50)

# Changing latent space and plotting it
# first getting max and min of latent space
from ccvae_analysis import get_namedlayer, lat_dimension
z = get_namedlayer('encoded', cvae.cvae, cvae.x_test, cvae.y_test, model = 'encoder')
dim = lat_dimension(z)

# Plotting axis change
from CVAE_3Dplots import construct_numvec, get_axis_change, plot_axis_change
lat_space = get_axis_change(0, 20, 2, cvae.decoder, 110, 2)
sliceview(lat_space)

plot_axis_change(0, 10, 2, cvae.decoder, 110)


index = []
for x in range(200):
    num = random.randint(0, 697)
    index.append(num)
lda_test = np.take(images, index, 0) # subsetting images
lda_test_label = [labels[i] for i in index] # subsetting labels
lda_label_onehot = to_categorical(lda_test_label) # encoding labels (not sure why)
lda_test = lda(encoder, lda_test, lda_label_onehot, lda_test_label)
lda_list lda_test[:10].index # get top ten dimensions
lda_list = lda_list.tolist()
# latent space grids (for important latent space dimensions found in above lda)
# traversal and ssim at end
de_list = plot_axis_change_grid(1, 10, 2, decoder, 8)
de_list = plot_axis_change_grid(4, 10, 2, decoder, 8)
# just ssim of all important dimensions
lda_imp = [8, 18, 42, 25, 39, 38, 34, 10, 0, 29, 44]
lat_ssim_diff_grid(decoder, 0, lda_imp)
lat_ssim_diff_grid(decoder, 4, lda_imp)
#from CVAEplots import plot_clusters
#plot_clusters(encoder, x_test, y_test, plot_labels_test, batch_size)

## Plotting digits as wrong labels 
## prbably female 2
#label = np.repeat(3, len(cvae.y_test))
#label_fake = to_categorical(cvae.label, num_classes=len(cvae.y_test[0]))
#reconstruction_plot(cvae.x_test, cvae.label_fake, cvae.cvae, slice= 2)
## probably male 1
#label = np.repeat(0, len(cvae.y_test))
#label_fake = to_categorical(cvae.label, num_classes=len(cvae.y_test[0]))
#reconstruction_plot(cvae.x_test, cvae.label_fake, cvae.cvae, slice= 2)
##from CVAEplots import lossplot
##lossplot(history)


#from CVAEplots import plot_latent_space
#plot_latent_space(1, 1.5, 8, decoder)

#from CVAEplots import latent_space_traversal
#latent_space_traversal(10, 1.5, decoder)

#from CVAEplots import plot_y_axis_change, plot_x_axis_change
#plot_y_axis_change(1, 10, 1.5, decoder)
#plot_x_axis_change(1, 10, 1.5, decoder)

#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.plot(sliceview(lat_space))
#ax2.plot(sliceview(lat_space))


#def reconstruction_plot(x_test, y_test, model, title, n=6):
#    ''' Reconstruct model outputs vs actual digits
#        n is number of digits, data is test (or train) image inputs, model is model'''
#    prediction = model.predict([x_test[:n+1], y_test[:n+1]])
#    import matplotlib.gridspec as gridspec
#    fig = plt.figure(figsize=(6, 6))
#    fig.suptitle(title, fontsize=10)
#    slice = [0, 1, 2, 4, 5, 8, 15]
#    gs1 = gridspec.GridSpec(6, 6)
#    gs1.update(wspace=0.000, hspace=0.000)
#    for i in range(1, n + 1):
#        # Display 1st line
#        ax = plt.subplot(6, n, i)
#        plt.imshow(x_test[i][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        # Display 2nd line
#        ax = plt.subplot(6, n, i + n)
#        plt.imshow(x_test[i + n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        # 3
#        ax = plt.subplot(6, n, i + n + n)
#        plt.imshow(x_test[i + n +n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        # 4
#        ax = plt.subplot(6, n, i + n + n +n)
#        plt.imshow(x_test[i + n +n+n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        #5
#        ax = plt.subplot(6, n, i + n + n +n+n)
#        plt.imshow(x_test[i + n +n+n+n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        #6
#        ax = plt.subplot(6, n, i + n + n +n+n+n)
#        plt.imshow(x_test[i + n +n+n+n+n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)

#    plt.show()

#prediction = cvae.predict([x_test, y_test])
#reconstruction_plot(x_test, y_test, 'Input data', cvae)
#reconstruction_plot(prediction, y_test, 'Reconstructions', cvae)
