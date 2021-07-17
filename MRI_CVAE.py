
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
import matplotlib.pyplot as plt
import nibabel as nib #reading MR images

filepath_df = pd.read_csv('Z:/PRONIA_data/Tables/pronia_full_niftis.csv')

mri_types =['wp0', # whole brain
            'wp1', # gm
            'wp2', # wm
            'mwp2', # modulated? wm
            'rp1', # gm mask
            'rp2'] # wm mask

niis = []
labels = []
num_subjs =10 # max 698

# Read in MRI image stacks
for i in range(num_subjs):
    row = filepath_df.iloc[i]
    nii_path = row['wp1']
    nii = nib.load(nii_path)
    nii = nii.get_fdata()
    nii = nii[:, 101:117, :] # gives 16 slices 36-51, ones with most variance (<80% similarity) (detailed in slice_variance.csv)
    labels.append(row['STUDYGROUP'])
    for j in range(nii.shape[1]):
        niis.append((nii[:,j,:]))
        
depth = len(nii[2])

# Prepare to crop images
#def crop_center(img,cropx,cropy):
#    y,x = img.shape
#    startx = x//2-(cropx//220) # // means floor division (rounds down to whole number)
#    starty = y//2-(cropy//2)    
#    return img[starty:starty+cropy,startx:startx+cropx]
# crop images in niis list
images = []
for i in range(len(niis)):
    img = niis[i]
    img = img[20:60, 50:90] # 12:108, 2:98 is whole brain
    images.append(img)
    
images = np.asarray(images) # shape num_subjs*121*121
# reshape to matrix in able to feed into network
images = images.reshape(-1, depth, 40, 40) # num_subjs,depth,121,121, (96,96)

### min-max normalisation to rescale between 1 and 0 to improve accuracy
m = np.max(images)
mi = np.min(images) 
images = (images - mi) / (m - mi)

# Normalise y for patient ages
#m = np.max(labels)
#mi = np.min(labels) 
#labels = (labels - mi) / (m - mi)

#from CVAE_3Dplots import plot_slices
#plot_slices(images)

## Pad images with zeros at boundaries so the dimenson is even and easier to downsample images by two while passing through model. Add in three rows and columns to make dim 176*176
#temp = np.zeros([num_subjs, depth,124,124])
#temp[:,:,3:,3:,] = images
#images = temp # dim now 5*2*124*124 # could replace temp with images 

# test train split (no labels for vae)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size=0.2, random_state=13)
train_label, test_label = y_train, y_test
y_train = to_categorical(y_train) # tuple num_patients * num_labels convert to onehot
y_test = to_categorical(y_test) # tuple num_patients * num_labels
# Y 0- healthy, 1- at risk, 2- recent depression, 4 - recent scz
 # Autoencoder variables
epochs = 50
batch_size = 8
#intermediate_dim = 124
latent_dim =50
n_y = y_train.shape[1] # 2
#n_y = 1 # For continuous variablw
n_x = x_train.shape[1] # 784
n_z = 2 # depth?
X, y = len(images[0][0][0]), len(images[0][0][0]) # should be 96, 96 messy fix though
inchannel = 1
origin_dim = 28*15 # why is this set


# Make a sampling layer, this maps the MNIST digit to latent-space triplet (z_mean, z_log_var, z), this is how the bottleneck is displayed. 
from keras import backend as K
def sampling(args):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean = 0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

# Build the model
from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Conv3DTranspose, BatchNormalization, SpatialDropout3D, Dropout
# Encoder
label = keras.Input(shape=(n_y, )) # shape of length y_train
encoder_inputs = keras.Input(shape=(depth, X, y, inchannel)) # it will add a None layer as batch size
x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(encoder_inputs) # relu turns negative values to 0
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x) # max pooling 
#x = layers.SpatialDropout3D(0.3)(x)
x = layers.Conv3D(64, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding ='same')(x) 
x = layers.SpatialDropout3D(0.3)(x)
x = layers.Conv3D(128, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x) 
#x = layers.SpatialDropout3D(0.3)(x)
#x = layers.Conv3D(256, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.Flatten()(x) # to feed into sampling function
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_sigma = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling, name='z')([z_mean, z_log_sigma])
z_label = concatenate([z, label], name='encoded') 
## initiating the encoder, it ouputs the latent dim dimensions
encoder = keras.Model([encoder_inputs, label], [z_mean, z_log_sigma, z_label, z], name="encoder")
encoder.summary()

#### Make the decoder, takes the latent keras
latent_inputs = keras.Input(shape=(latent_dim + n_y),) # changes based on depth 
x =  layers.Dense(2*5*5*128, activation='relu')(latent_inputs)
x = layers.Reshape((2, 5, 5, 128))(x)
#x = layers.UpSampling3D((2,2,2))(x)
#x = layers.Conv3DTranspose(128, (3, 3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling3D((2,2,2))(x)
x = layers.Conv3DTranspose(64, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.SpatialDropout3D(0.3)(x)
x = layers.UpSampling3D((2,2,2))(x)
#x = layers.SpatialDropout3D(0.2)(x)
x = layers.Conv3DTranspose(32, (3, 3, 3), activation="relu",  padding="same")(x)
x = layers.UpSampling3D((2,2,2))(x)
#x = layers.SpatialDropout3D(0.3)(x)
decoder_outputs = layers.Conv3DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# Initiate decoder
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# Instantiate and fit VAE model, outputs only z_label
outputs = decoder(encoder([encoder_inputs, label])[2])
cvae = keras.Model([encoder_inputs, label], outputs, name='cvae')
cvae.summary()

# use a custom loss function, this includes a KL divergence regularisation term which ensures that z is close to normal (0 mean, 1 sd)
reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
reconstruction_loss *= origin_dim # how does this impact model
reconstruction_loss = K.mean(reconstruction_loss) # mean to avoid incompatible shape error
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
cvae_loss = reconstruction_loss + kl_loss # mean was worse

# Add loss and compile cvae model
cvae.add_loss(cvae_loss)
opt = keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.009)
cvae.compile(optimizer=opt)

# Tensorboard
from keras.callbacks import TensorBoard
import datetime
log_dir = "C:/Users/Mischa/sophie/29_06_21_onwards/MRI_CVAE" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Adding early stopping
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# fit the data 
history = cvae.fit([x_train, y_train], x_train, epochs=epochs, batch_size=batch_size, validation_data = ([x_test, y_test], x_test), verbose = 2, callbacks=[tensorboard_callback, es_callback])

####################
from CVAE_3Dplots import reconstruction_plot
reconstruction_plot(x_test, y_test, cvae)

#from ccvae_analysis import structural_sim_data, var_boxplot, variation_summary
# var_boxplot calls structural_sim_data
#var_boxplot(x_test, y_test, cvae)
# var_summary calls var_boxplot
#variation_summary(x_test)

#import os
#os.system("python %s %d %s" % (MRI_CVAE, x_test, y_test))

# # # # # MODEL END ANALYSIS START # # # # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # 

#from ccvae_analysis import get_namedlayer
#encoded_output = get_namedlayer('encoded', cvae, x_train, y_train)

### Looking at variation between predictions and x_test sets (predictions are all very similar, 99%)
#from ccvae_analysis import structural_sim_data, var_boxplot, var_summary
## var_boxplot calls structural_sim_data
#var_boxplot(x_test, y_test, cvae)
## var_summary calls var_boxplot
#var_summary(x_test)

######## Linear Discriminant analysis ##
#from ccvae_analysis import lda
#lda(encoder, x_train, y_train, train_label)

#from CVAE_3Dplots import plot_clusters
#plot_clusters(encoder, x_test, y_test, test_label, batch_size = 16)
#plot_clusters(encoder, x_train, y_train, train_label, batch_size = 16)

##from CVAEplots import plot_clusters
##plot_clusters(encoder, x_test, y_test, plot_labels_test, batch_size)

## Plotting digits as wrong labels 
## prbably female 2
#label = np.repeat(3, len(y_test))
#label_fake = to_categorical(label, num_classes=len(y_test[0]))
#reconstruction_plot(x_test, label_fake, cvae, slice= 2)
## probably male 1
#label = np.repeat(0, len(y_test))
#label_fake = to_categorical(label, num_classes=len(y_test[0]))
#reconstruction_plot(x_test, label_fake, cvae, slice= 2)
#from CVAEplots import lossplot
#lossplot(history)

## Plotting variation between latent space
#img_it = 0
#for i in range(0, 250):
#    z1 = (((i / (250-1)) * max_z)*2) - max_z
#    z_ = [0, z1] # This is where the x axis changes
#    vec = construct_numvec(1, z_)
#    decoded = decoder.predict(vec)
#    ax = plt.subplot(10, 1, 1 + img_it)
#    img_it +=1
#    plt.imshow(decoded.reshape(96, 96), cmap = plt.cm.gray)
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)  
#plt.show()

## Plots ###########################################################################


# for i in range(1, n+1):
#    # display reconstruction learning
#    ax = plt.subplot(1, n, i)
#    plt.imshow(intermediate_output[i].reshape((7, 7*8)).T)
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()




#from CVAEplots import plot_latent_space
#plot_latent_space(1, 1.5, 8, decoder)

#from CVAEplots import latent_space_traversal
#latent_space_traversal(10, 1.5, decoder)

#from CVAEplots import plot_y_axis_change, plot_x_axis_change
#plot_y_axis_change(1, 10, 1.5, decoder)
#plot_x_axis_change(1, 10, 1.5, decoder)


###############################################################

#for i in range(n_z+n_y):
#	tmp = np.zeros((1,n_z+n_y))
#	tmp[0,i] = 1
#	generated = decoder.predict(tmp)
#	file_name = './img' + str(i) + '.jpg'
#	print(generated)
#	imsave(file_name, generated.reshape((28,28)))
#	sleep(0.5)

# this loop prints a transition through the number line

#pic_num = 0
#variations = 30 # rate of change; higher is slower
#for j in range(n_z, n_z + n_y - 1):
#	for k in range(variations):
#		v = np.zeros((1, n_z+n_y))
#		v[0, j] = 1 - (k/variations)
#		v[0, j+1] = (k/variations)
#		generated = decoder.predict(v)
#		pic_idx = j - n_z + (k/variations)
#		file_name = './transition_50/img{0:.3f}.jpg'.format(pic_idx)
#		imsave(file_name, generated.reshape((28,28)))
#		pic_num += 1
		
