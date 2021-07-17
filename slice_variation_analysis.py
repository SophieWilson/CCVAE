import numpy as np
from keras.utils import to_categorical
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
import nibabel as nib #reading MR images

## Image slice analysis to see whats the most variation # # # #
def slice_analysis(images):
    ''' wrapper function for slice analysis, input images as variable '''
    def structural_sim_data(data):
        ''' Returns a list of length depth, containing samples*samples-samples variances '''
        from skimage.metrics import structural_similarity as ssim
        print(len(data[0]))
        results = []   
        temp = []
        count = 0
        for k in range(int(len(data[0]))): # looping through slice depth
            for i in range(len(data)): # looping through patients in x 60
                for j in range(len(data)): # looping again to compare 60
                    if (i == j):
                        continue
                    else:
                        (score, diff) = ssim(data[i][k], data[j][k], full=True)
                        temp.append(score)
                        #print(len(temp))
            results.append(temp)
            temp = [] 
            print(count)
            count+=1
        return results

    image_var = structural_sim_data(images)
    import statistics
    slice_var = []
    counter = 0
    for i in range(len(images[0])):
        counter += 1
        slice_var.append([i, statistics.mean(image_var[i])]) # taking averages of the variations
        if (counter%10 == 0):
            print(counter)

    slice_df = pd.DataFrame(slice_var)
    slice_df.to_csv('C:/Users/Mischa/Documents/Uni Masters/Diss project/Practise/MRI_cvae/50_101.csv')
    return(slice_df)

#### Latent space variation analysis # # # 

def latent_ssim_analysis(label, num_recon, max_z, decoder, latent_dim):

    def latent_space_var():
        ''' this is messy but it does work. gives reconstructions for images across the latent space. num_recon is number of reconstructions you want to test over, max_z is maximum and minimum latent space dim. '''
        #from CVAE_3Dplots import construct_numvec
        decoded_list = [[] for x in range(latent_dim)]
        for i in range(latent_dim): # looping through dimensions
            z_ = [0] * i
            for j in range(0, num_recon): # looping through number of images
                z1 = (((j / (num_recon-1)) * max_z)*2) - max_z
                z_.append(z1)
                vec = construct_numvec(label, z_)
                decoded = decoder.predict(vec) # 1, 16, 96, 96, 1
                decoded = decoded[0,:,:,:,0] # slice depth, 96, 96
                decoded_list[i].append(decoded)
                z_.pop()
        return decoded_list

    def structural_sim_latent(data):
        ''' Returns a list of length depth, containing samples*samples-samples variances, dont think i have to seperate the slices as it seems to cope with multiple'''
        from skimage.metrics import structural_similarity as ssim
        print(len(data[0]))
        results = []   
        diff_list = []
        temp = []
        count = 0
        for k in range(int(len(data))): # looping through latent dim 
            for i in range(len(data[0])): # looping through recons
                for j in range(len(data[0])): # looping through recon again 
                    if (i == j):
                        continue
                    else:
                        (score, diff) = ssim(data[k][i], data[k][j], full=True)
                        temp.append(score)
                        #print(len(temp))
            results.append(temp)
            diff_list.append(diff)
            temp = [] 
            print(count)
            count+=1
        return results, diff_list

    lat_var = latent_space_var()
    # lat_var is of shape 30, 10, (16, 96, 96), 30 dimensions, 10 recons 
    latent_ssim, diff_list = structural_sim_latent(lat_var)
    latent_slice_var = []
    counter = 0
    import statistics
    for i in range(len(latent_ssim)):
        print(latent_ssim)
        counter += 1
        latent_slice_var.append([i, statistics.mean(latent_ssim[i])])
        #if (counter%10 == 0):
            #print(counter)

    latent_df = pd.DataFrame(latent_slice_var)
    latent_df = latent_df.sort_values(by = [1], ascending = True)
    return latent_df, diff_list

#lat_df, diff = latent_ssim_analysis(0, 10, 4, decoder, 50)

#def latent_space_var(label, sides, max_z, decoder, latent_dim):
#    decoded_list = [[] for x in range(latent_dim)]
#    for i in range(latent_dim): # looping through dimensions
#        z_ = [0] * i
#        for j in range(0, sides): # looping through number of images
#            z1 = (((j / (sides-1)) * max_z)*2) - max_z
#            z_.append(z1)
#            vec = construct_numvec(label, z_)
#            decoded = decoder.predict(vec)
#            decoded = decoded[:,:,:,:,0]
#            decoded_list[i].append(decoded)
#    return decoded_list


