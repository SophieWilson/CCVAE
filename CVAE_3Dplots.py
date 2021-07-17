
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import pandas as pd
## Plots
## All of the plotting functions for CVAE  models in one place as they're clogging up the actual script, example of how to call below each function ##
## These plots are adapted for 3D input, to show slices ##
# Functions in this script:
#   plot_clusters, digit_grid, reconstruction_plot, lossplot, construct_numvec, plot_latent_space, latent_space_traversal, plot_x_axis_change, plot_y_axis_change, plot_lda_clusters

def plot_clusters(encoder, x_test, y_test, labels, batch_size):
    ''' Display how the latent space clusters the digit classes '''
    x_test_encoded, _, _, _ = encoder.predict([x_test, y_test], batch_size=batch_size)
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=labels)
    plt.colorbar()
    fig.suptitle('Input images plotted in latent space', fontsize=10)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
#plot_clusters(encoder, [x_test, y_test], plot_labels_test, batch_size)


# Plotting reconstruction vs actual
def reconstruction_plot_old(x_test, y_test, model, n=6):
    ''' Reconstruct model outputs vs actual digits
        n is number of digits, data is test (or train) image inputs, model is model'''
    prediction = model.predict([x_test[:n+1], y_test[:n+1]])
    fig = plt.figure(figsize=(20, 4))
    fig.suptitle('Reconstructions vs input digits', fontsize=10)
    slice = [0, 1, 2, 4, 5, 8, 15]
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        ax.set_title(y_test[i])
        plt.imshow(prediction[i][slice[i]].reshape(prediction.shape[2],prediction.shape[2]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def reconstruction_plot(x_test, y_test, model, n=6):
    ''' Reconstruct model outputs vs actual digits
        n is number of digits, data is test (or train) image inputs, model is model'''
    prediction = model.predict([x_test[:n+1], y_test[:n+1]])
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(6, 6))
    #fig.suptitle(title, fontsize=10)
    slice = [0, 1, 2, 4, 5, 8, 15]
    gs1 = gridspec.GridSpec(6, 6)
    gs1.update(wspace=0.000, hspace=0.000)
    for i in range(1, n + 1):
        # Display 1st line
        ax = plt.subplot(6, n, i)
        plt.imshow(x_test[i][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Display 2nd line
        ax = plt.subplot(6, n, i + n)
        plt.imshow(x_test[i + n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # 3
        ax = plt.subplot(6, n, i + n + n)
        plt.imshow(x_test[i + n +n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # 4
        ax = plt.subplot(6, n, i + n + n +n)
        plt.imshow(x_test[i + n +n+n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #5
        ax = plt.subplot(6, n, i + n + n +n+n)
        plt.imshow(x_test[i + n +n+n+n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #6
        ax = plt.subplot(6, n, i + n + n +n+n+n)
        plt.imshow(x_test[i + n +n+n+n+n][slice[i]].reshape(x_test.shape[2], x_test.shape[2])) # sample i, input slice
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

prediction = cvae.predict([x_test, y_test])
reconstruction_plot(x_test, y_test, 'Input data', cvae)
reconstruction_plot(prediction, y_test, 'Reconstructions', cvae)

def plot_slices_vert(x_test, n = 6 ):
    ''' working, x_test is data, n is number displayed '''
    fig = plt.figure(figsize=(5, 13))
    fig.suptitle('Input slices', fontsize=10)
    slices = [1, 3, 6, 9, 12, 15]
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(n, 1, i)
        plt.imshow(x_test[1][slices[i-1]]) # sample i, input slice
        plt.gray()
        ax.set(ylabel = slices[i-1])
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
    plt.show()


def plot_ssim_diff(data, diff_slice, n = 6):
    ''' working, x_test is data, n is number displayed, this works with output of latent_ssim_analysis to make one plot '''
    fig = plt.figure(figsize=(5, 13))
    fig.suptitle('Latent space differences', fontsize=10)
    slice = [0, 3, 6, 9, 12, 15]
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(n, 1, i)
        plt.imshow(data[diff_slice][slice[i-1]]) # sample i-1 (to start from 0), input slice
        plt.gray()
        #ax.set(ylabel = slice)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax.set(ylabel = slice[i-1])
    plt.show()



def plot_slices(x_test, n = 15):
    ''' working '''
    fig = plt.figure(figsize=(20, 3))
    fig.suptitle('Input slices', fontsize=10)
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[1][i]) # sample i, input slice
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def plot_data(x_test, n=9):
    fig = plt.figure(figsize=(20, 2))
    fig.suptitle('Input digits', fontsize=15)
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        quard = int(math.sqrt(x_test.shape[1])) # sqrt of x dimension
        plt.imshow(x_test[i].reshape(quard, quard))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# loss plot
def lossplot(history):
    ''' Plotting loss as a line graph, history is the variable saved in model.fit() '''
    loss_values = history.history['loss']
    val_loss = history.history['val_loss']
    #reconstruction_loss = history.history['reconstruction_loss']
    #kl_loss = history.history['kl_loss']
    epochs = range(1, len(loss_values)+1)
    fig = plt.figure()
    fig.suptitle('Training loss plot', fontsize=10)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss, label = 'Val_loss')
    #plt.plot(epochs, kl_loss, label='KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
#lossplot(history)

# Reconstructing specific digits
def construct_numvec(label, z = 3, n_z = 50, n_y = 5):
    ''' make number vector, its called in plot_latent_space, must change n_z and n_y values  here if you want to fix a plot '''
    out = np.zeros((1, n_z + n_y))
    out[:, label + n_z] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        return(out)
    
# Plotting one label
#sample_3 = construct_numvec(3)
#plt.figure(figsize=(3, 3))
#plt.imshow(decoder.predict(sample_3).reshape(28,28), cmap = plt.cm.gray)
#plt.show()


def plot_latent_space(label, max_z, sides, decoder):
    ''' Plotting latent space with respect to specific numbers 
        dig = 1
        sides = 8
        max_z = 1.5 '''
    img_it = 0 
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Latent space of specific digit', fontsize=10)
    for i in range(0, sides):
        z1 = (((i / (sides-1)) * max_z)*2) - max_z
        for j in range(0, sides):
            z2 = (((j / (sides-1)) * max_z)*2) - max_z
            z_ = [z1, z2]
            vec = construct_numvec(label, z_)
            decoded = decoder.predict(vec)
            ax = plt.subplot(sides, sides, 1 + img_it)
            img_it +=1
            plt.imshow(decoded[j].reshape(96, 96), cmap = plt.cm.gray)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
    plt.show()


# changing axis
def get_axis_change(label, sides, max_z, decoder, latent_dim, slice_num):
    ''' feed into sliceview to get a scroll of latent space '''
    z_ = [0] * latent_dim
    dec_list = []
    for i in range(0, sides):
        z1 = (((i / (sides-1)) * max_z)*2) - max_z
        z_.append(z1) # This is where the axis changes (right now its first, try to change that)
        vec = construct_numvec(label, z_)
        decoded = decoder.predict(vec)
        decoded = decoded.reshape(16, 40, 40)
        dec_list.append(decoded)
    dec_list = [x[slice_num] for x in dec_list]
    return dec_list

#def plot_axis_change(label, sides, max_z, decoder, latent_dim):
#    ''' Plotting x axis change
#    sides = number of recons
#        have been using 1, 10, 1.5 for inputs, gives strange outputs '''
#    #from CVAE_3Dplots import construct_numvec
#    img_it = 0
#    fig = plt.figure(figsize = (4, 20))
#    fig.suptitle('Varying axis', fontsize=10)
#    z_ = [0] * latent_dim
#    for i in range(0, sides):
#        z1 = (((i / (sides-1)) * max_z)*2) - max_z
#        z_.append(z1) # This is where the axis changes (right now its first, try to change that)
#        vec = construct_numvec(label, z_)
#        decoded = decoder.predict(vec)
#        ax = plt.subplot(10, 1, 1 + img_it)
#        img_it +=1
#        plt.imshow(decoded[0][15].reshape(40, 40), cmap = plt.cm.gray)
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)  
#        z_.pop()
#    plt.show()
#plot_axis_change(1, 10, 2, decoder, 2)

def plot_axis_change(label, sides, max_z, decoder, latent_dim):
    ''' Plotting x axis change
    sides = number of recons
        have been using 1, 10, 1.5 for inputs, gives strange outputs '''
    #from CVAE_3Dplots import construct_numvec
    import matplotlib.gridspec as gridspec
    img_it = 0
    fig = plt.figure(figsize = (20, 8))
    fig.suptitle('Varying axis', fontsize=10)
    gs1 = gridspec.GridSpec(5, 10)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    z_ = [0] * latent_dim
    #plt.xlabel('one')
    #plt.ylabel('work')
    for j in range(6):
        slice = j * 3
        for i in range(0, 10):
            z1 = (((i / (sides-1)) * max_z)*2) - max_z
            z_.append(z1) # This is where the axis changes (right now its first, try to change that)
            vec = construct_numvec(label, z_)
            decoded = decoder.predict(vec)
            ax = plt.subplot(6, 10, 1 + img_it)
            img_it +=1
            plt.imshow(decoded[0][slice].reshape(40, 40), cmap = plt.cm.gray)

            #ax.get_xaxis().set_ticklabels([])
            #ax.get_yaxis().set_ticklabels([])
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

            ax.set(ylabel = slice, xlabel = (round(z1, 1)))
            z_.pop()

    plt.show()


def plot_axis_change_grid(label, sides, max_z, decoder, latent_dim):
    ''' Plotting x axis change
    sides = number of recons
        have been using 1, 10, 1.5 for inputs, gives strange outputs '''
    #from CVAE_3Dplots import construct_numvec
    import matplotlib.gridspec as gridspec
    img_it = 0
    fig = plt.figure(figsize = (8, 10))
    #fig.suptitle('Varying axis', fontsize=10)
    gs1 = gridspec.GridSpec(5, 10)
    gs1.update(wspace=0.1, hspace=0.1) # set the spacing between axes. 
    z_ = [0] * latent_dim
    #plt.xlabel('one')
    #plt.ylabel('work')
    for j in range(6): # looping through slices
        slice = j * 3
        for i in range(0, 11): # looping through lat dim
            z1 = (((i / (sides-1)) * max_z)*2) - max_z
            print(z1)
            z_.append(z1) # This is where the axis changes (right now its first, try to change that)
            vec = construct_numvec(label, z_)
            decoded = decoder.predict(vec)
            ax = plt.subplot(6, 11, 1 + img_it)
            img_it +=1
            if (i == 0):
                first = decoded[0][slice].reshape(40, 40)
            if (i == 10):
                last = decoded[0][slice].reshape(40, 40)
            if (img_it % 11 == 0):
                score, diff = ssim(first, last, full = True)
                plt.imshow(diff.reshape(40, 40), cmap = plt.cm.gray)
                plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            else:
                plt.imshow(decoded[0][slice].reshape(40, 40), cmap = plt.cm.gray)

            #ax.get_xaxis().set_ticklabels([])
            #ax.get_yaxis().set_ticklabels([])
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

            #ax.set(ylabel = slice, xlabel = (round(z1, 1)))
            z_.pop()

    plt.show()

#de_list = plot_axis_change_grid(1, 10, 2, decoder, 8)

def lat_ssim_diff_grid(decoder, label, lat_dim):
    lat_dim = lat_dim
    fig = plt.figure(figsize = (8, 6))
    img_it = 0
    for j in range(6): # looping through slices
        slice = j*3
        for i in range(0, 8): # looping through lat dim
            z_neg  = [0] * lat_dim[i]
            z_pos = [0] * lat_dim[i]
            z_neg.append(-2.0)
            z_pos.append(2.0)
            vec_n = construct_numvec(label, z_neg)
            vec_p = construct_numvec(label, z_pos)
            decoded_n = decoder.predict(vec_n)
            decoded_p = decoder.predict(vec_p)
            decoded_n = decoded_n[:,:,:,:,0]
            decoded_p = decoded_p[:,:,:,:,0]
            ax = plt.subplot(6, 8, 1 + img_it)
            img_it +=1
            neg = decoded_n[0][slice].reshape(40, 40)
            pos = decoded_p[0][slice].reshape(40, 40)
            score, diff = ssim(neg, pos, full = True)
            plt.imshow(diff.reshape(40, 40), cmap = plt.cm.gray)
            #ax.set(ylabel = slice, xlabel = lat_dim[i])
            #ax.get_xaxis().set_ticklabels([])
            ##ax.get_yaxis().set_ticklabels([])
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            z_neg.pop()
            z_pos.pop()

    plt.show()
lat_ssim_diff_grid(decoder, 0, list2)


def diff_gridplot(data,  n = 6):
    ''' working, x_test is data, n is number displayed, this works with output of latent_ssim_analysis to make one plot '''
    fig = plt.figure(figsize=(5, 13))
    fig.suptitle('Latent space differences', fontsize=10)
    #slice = [0, 3, 6, 9, 12, 15]
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(n, 1, i)
        plt.imshow(data[i-1]) # sample i-1 (to start from 0), input slice
        plt.gray()
        #ax.set(ylabel = slice)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax.set(ylabel = i-1)
    plt.show()
# Plotting digits as wrong labels 
# setting fake label
#label = np.repeat(7, 10000)
#label_fake = to_categorical(label, num_classes=10)

def plot_lda_cluster(X, y, title, label_dict, sklearn_lda):
    ''' Plots LDA clusters within the subspace, LDA function must be run for this plot to work '''
    #X_lda = LDA(encoder, train_label, label_dict) # run the LDA analysis

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,5),('v','^', 's', 'o'),('blue','orange', 'green', 'red')):

        plt.scatter(x=X[:,0][y == label],
                    y=X[:,1][y == label] * -1, # flip the figure
                    color=color,
                    alpha=0.5,
                    label = label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.2)
    plt.suptitle(title)
    plt.title('Explained variance of LD1+LD2 = ' + str(round((sklearn_lda.explained_variance_ratio_[1] + sklearn_lda.explained_variance_ratio_[0]), 2)), fontsize=10)
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")
    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    x1 = np.array([np.min(X[:,0], axis=0), np.max(X[:,0], axis=0)])
    # Trying to add lines in, not working
    for i, c in enumerate(['purple','blue','red', 'green']):
        b, w1, w2, w2 = sklearn_lda.intercept_[i], sklearn_lda.coef_[i][0], sklearn_lda.coef_[i][1], sklearn_lda.coef_[i][2]
        y1 = -(b+x1*w1)/w2    
        #plt.plot(x1,y1,c=c) 

    plt.grid()
    plt.tight_layout
    plt.show()

### Making histogram of dimensions (3 dim as 4 groups)
def lda_densityplot(X_lda, y, label, sklearn_lda):#
    ''' x_lda is lda analysis on encoder output, y is the label vector, label is the group description (STUDYGROUP, SEX etc.), this must be a string '''
    import seaborn as sns 
    df = pd.DataFrame(X_lda)
    df['label'] = y
    fig, axes = plt.subplots(1, 3, figsize=(12,6))
    for i in range(len(X_lda[0])):
        sns.kdeplot(x=df.iloc[:,i], hue = df['label'], palette='tab10', fill = True, ax = axes[i])    
        title = 'Dimension ' + str(i+1) + ', explained variance:' + str(round(sklearn_lda.explained_variance_ratio_[i], 2))
        axes[i].set_title(title)
    plt.show()

#def lda_biplot():


def sliceview(volume, axis=0, rot=1, show=True):
    """View a 3d image volume with mousewheel scrolling to progress through slices.
    Optional arguments:
    axis: which axis the mousewheel controls.
    rot: how many times to rotate the visible plane in 90deg turns
    show: whether or not to instantly show the plot"""
    fig, ax = plt.subplots()

    # optionally rotate the front facing view by an integer amount of 90deg turns
    if rot != 0:
        not_axis = [0,1,2]
        not_axis.remove(axis)
        volume = np.rot90(volume, k=rot, axes=not_axis)
    ax.volume = volume

    if axis == 0:
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index], cmap='gray')
    elif axis == 1:
        ax.index = volume.shape[1] // 2
        ax.imshow(volume[:,ax.index], cmap='gray')
    elif axis == 2:
        ax.index = volume.shape[2] // 2
        ax.imshow(volume[:,:,ax.index], cmap='gray')
    else:
        raise Exception("Axis must be 0, 1, or 2")

    # wrapper function to encode axis argument into process_mwheel:
    def process_mwheel_along_axis(*args):
        return process_mwheel(*args, axis=axis)

    fig.canvas.mpl_connect('scroll_event', process_mwheel_along_axis)
    ax.anno = ax.annotate(f'z={ax.index}/{ax.volume.shape[axis]-1}',
            xy=(3, 1), xycoords='data',
            xytext=(0.05, 0.95), textcoords='axes fraction',
            c='white',
            horizontalalignment='left', verticalalignment='top')
    if show:
        plt.show()

def process_mwheel(event, axis=0):
    """mousewheel event handler for sliceview function"""
    fig = event.canvas.figure
    ax = fig.axes[0]
    old_index = ax.index

    if event.button == 'down':
        ax.index = np.maximum(ax.index-1, 0)
    elif event.button == 'up':
        ax.index = np.minimum(ax.index+1, ax.volume.shape[axis]-1)

    if axis == 0:
        ax.images[0].set_array(ax.volume[ax.index])
    elif axis == 1:
        ax.images[0].set_array(ax.volume[:,ax.index])
    elif axis == 2:
        ax.images[0].set_array(ax.volume[:,:,ax.index])

    ax.anno.remove()
    ax.anno = ax.annotate(f'z={ax.index}/{ax.volume.shape[axis]-1}',
            xy=(3, 1), xycoords='data',
            xytext=(0.05, 0.95), textcoords='axes fraction',
            c='white',
            horizontalalignment='left', verticalalignment='top')
    fig.canvas.draw()

#slice = get_axis_change(1, 20, 2, decoder, 110, 2)
#sliceview(slice)

## t-SNE on lat space
def tsne(): # NOT FINISHED DONT IMPORT
    import time
    time_start = time.time()
    from sklearn.manifold import TSNE
    import seaborn as sns

    z_mean_pred, z_sig, z_label_pred, z_pred = encoder.predict([images, to_categorical(labels)], batch_size=16)
    #y = np.array(train_label)
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=10000, learning_rate=10)
    tsne_results = tsne.fit_transform(z_pred)

    ## pca reduction
    from sklearn.decomposition import PCA
    data = pd.DataFrame(z_pred)
    pca = PCA(n_components = 50)
    pca_result = pca.fit_transform(data)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    tsne_pca= tsne.fit_transform(pca_result)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    # lda reduction
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    sklearn_lda = LinearDiscriminantAnalysis()
    y = np.array(train_label)
    z_pred = pd.DataFrame(z_pred)
    X_lda = sklearn_lda.fit_transform(data, labels)
    #X_lda = sklearn_lda.transform(z_pred)
    tsne_lda= tsne.fit_transform(X_lda)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    # plot t-SNE
    data['tsne-2d-one'] = tsne_results[:,0]
    data['tsne-2d-two'] = tsne_results[:,1]
    data['y'] = labels
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("tab10", 4),
        data=data,
        legend="full",
        alpha=0.7
    )
    plt.show()
    # t-sne on dataset (images)
    dat =pd.DataFrame(images).values
    tsne_results2 = tsne.fit_transform(dat)
