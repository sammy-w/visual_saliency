import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
import progressbar
import pandas as pd

proximity_array = []
 #show progressbar
bar = progressbar.ProgressBar(maxval=810, \
widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

#run script for video 33
for i in range(1,811):

    test = np.load(f'./deep_gaze/density_maps/{i}.npz') #compressed dictionary object per frame
    #np.shape(test['arr_0']) #(x,y) values, with density across 2d space

    #apply thresholding? False/True coding for areas above density threshold. Set quantile at 95% of data points (highest density)
    thresholded_coordinates = (test['arr_0'] > np.quantile(test['arr_0'],0.95)).nonzero() #returns tuple of x,y coordinates for top 5% dense areas
    #thresholded = test['arr_0'] > np.quantile(test['arr_0'],0.95)
    #thesholded = np.nonzero(thresholded)

    ##calculate euclidian distances between gaze coordinates (pos_x and pos_y) per time point
    euclidean_distances = distance.pdist(thresholded_coordinates, 'euclidean') 
    #calculate sum of distances and proximity measure (inverse)
    avg_distance = np.nanmean(euclidean_distances)
    proximity_measure = 1/avg_distance
    #append to list
    proximity_array.append(proximity_measure)

    bar.update(i)

#create dataframe with time and measure
time = np.linspace(0,27,810)
column_values = ["Time","Concentration"]
matrix = np.vstack([time,proximity_array])
matrix = np.transpose(matrix)
np.shape(matrix)
data = pd.DataFrame(matrix,columns=column_values)

#show salinecy concentration over time
plt.plot(data['Time'],data['Concentration'])
plt.show()

data.to_csv('video33.csv')

#visualize data
empty_frame = np.zeros((1080, 1920,3))
plt.gca().imshow(empty_frame, alpha=0.2)
plt.scatter(thresholded_coordinates)
#m = plt.gca().matshow(test['arr_0'], alpha=0.5, cmap=plt.cm.RdBu)
m = plt.gca().matshow(test['arr_0'] > np.quantile(test['arr_0'],0.95), alpha=0.5, cmap=plt.cm.RdBu)
plt.colorbar(m)
plt.show()

