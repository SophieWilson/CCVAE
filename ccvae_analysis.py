import MRI_CVAE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
# # # # # MODEL END ANALYSIS START # # # # # # 

# This will extract all layers outputs from the model
def get_layers(cvae, x_train, y_train):
    ''' gets output for all layers in the model 
    get_layers(cvae, x_train, y_train) '''
    extractor = keras.model(inputs = cvae.inputs, outputs=[layer.output for
    layer in cvae.layers])
    extractor = keras.model([x_train, y_train])

# This will get one named layer from the model
def get_namedlayer(layer_name, cvae, x_train, y_train, model = 'encoder'):
    ''' gets one named layer of model
    get_namedlayer('encoded', cvae, x_train, y_train) '''
    layer = layer_name
    intermediate_layer_model = keras.Model(inputs=[cvae.inputs], outputs=[cvae.get_layer(model).get_layer(layer).get_output_at(0)])
    intermediate_output = intermediate_layer_model.predict([x_train, y_train]) # intermediate output is label, 1503 dense, reshape to 
    return intermediate_output

def lat_dimension(z):
    list = []
    for i in range(len(z[1])):
        temp = [x[i] for x in z]
        tup = [max(temp), min(temp)]
        list.append(tup)
    return(list)

## Looking at variation between predictions and x_test sets (predictions are all very similar, 99%)

def structural_sim_data(data):
    ''' Returns a list of length depth, containing samples*samples-samples variances '''
    from skimage.metrics import structural_similarity as ssim
    results = []   
    temp = []
    for k in range(int(len(data[0]))): # looping through slice depth
        for i in range(len(data)): # looping through patients in x 60
            for j in range(len(data)): # looping again to compare 60
                if (i == j):
                    continue
                else:
                    (score, diff) = ssim(data[i][k], data[j][k], full=True)
                    temp.append(score)
        results.append(temp)
        temp = []
    return results

 # # Comparing results and actual data
def var_boxplot(x_test, y_test, cvae):
    var_boxplot.x_test_results = structural_sim_data(x_test)

    prediction = cvae.predict([x_test, y_test])
    prediction = prediction[:,:,:,:,0]
    var_boxplot.prediction_results = structural_sim_data(prediction)
    import seaborn as sns
    import matplotlib.pyplot as plt
    ax = sns.boxplot(data = [var_boxplot.x_test_results, var_boxplot.prediction_results])
    ax.set(xlabel = ['Input data', 'Reconstructions'])
    plt.show()

def variation_summary(x_test):
    ''' calls from var_boxplot '''
    import statistics
    prediction_results = var_boxplot.prediction_results
    x_test_results = var_boxplot.x_test_results
    slice_var = []
    for i in range(len(x_test_results)):
        slice_var.append([i, statistics.mean(x_test_results[i])])
    slice_var_pred = []
    for i in range(len(prediction_results)):
        slice_var_pred.append([i, statistics.mean(prediction_results[i])])
    #var_df = pd.DataFrame(slice_var, slice_var_pred)
    #return var_df
    slice_var_2 = [x[1] for x in slice_var]
    slice_var_pred_2 = [x[1] for x in slice_var_pred]
    print(statistics.mean(slice_var_2), statistics.mean(slice_var_pred_2))

####### Linear Discriminant analysis ##
def lda(encoder, x_train, y_train, train_label):
    ''' total LDA analysis, outputs plots and can return a variable too if needed
    lda(encoder, x_train, y_train, train_label) '''
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    z_mean_pred, z_sig, z_label_pred, z_pred = encoder.predict([x_train, y_train], batch_size=16)
    sklearn_lda = LinearDiscriminantAnalysis()
    y = np.array(train_label)
    z_pred = pd.DataFrame(z_pred)
    sklearn_lda = sklearn_lda.fit(z_pred, y)
    X_lda = sklearn_lda.transform(z_pred)
    print(len(X_lda[0]))
    score = sklearn_lda.score(z_pred, y)
    print('accruacy', score)
    label_dict = {1: 'Healthy', 2: 'At risk of SCZ', 3:'Depression', 4:'SCZ'}

    #from CVAE_3Dplots import lda_densityplot
    lda_densityplot(X_lda, y, 'STUDYGROUP', sklearn_lda)

    #from CVAE_3Dplots import plot_lda_cluster
    plot_lda_cluster(X_lda, y, '', label_dict, sklearn_lda)


    importance = pd.DataFrame(sklearn_lda.scalings_)
    print(sklearn_lda.explained_variance_ratio_)
    print(importance.shape)
    print(sklearn_lda.confusion_matrix) 
    exp_var = sklearn_lda.explained_variance_ratio_.tolist()
    importance.loc[len(importance)] = exp_var
    importance = importance.abs() # removing all negative numbers
    importance['totals'] = (importance[0] * importance.iloc[50,0]) +  (importance[1] * importance.iloc[50,1]) +  (importance[2] * importance.iloc[50,2])
    importance = importance.sort_values(by = ['totals'], ascending = False)
    return importance

#lda(encoder, x_test, y_test, test_label)     
#lda(encoder, x_train, y_train, train_label)     

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
