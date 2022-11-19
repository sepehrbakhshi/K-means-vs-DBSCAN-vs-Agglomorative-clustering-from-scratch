import time as time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import randomcolor
import time
from sklearn import preprocessing
class MyDBSCAN(object):

    def __init__(self, min_points, eps, data ):
       self.eps = eps
       self.min_points = min_points
       self.dimensions  = 3
       self.data = data
       self.labels =  np.zeros(shape=(self.data.shape[0],1))
       self.error = 0
       self.neighbours = []
       self.cluster_counter = 1
     #calculating euclidean distance
    """
    def euclidean_distance2(self,mydata,center):
        eu_pow = 0
        eu_distance = 0
        eu_sum = 0
        eu_pow = (mydata - center)
        #print(eu_pow[0])
        eu_pow = eu_pow**2
        eu_pow = np.array(eu_pow)
        print(eu_pow)
        #eu_sum = eu_pow.sum(axis=1)
        eu_distance = np.sqrt(eu_pow)
        return eu_distance
    """
    def fit(self):
        neighbours = []
        for i in range(0, self.data.shape[0]):
             if(self.labels[i] != 0):
                 continue
             neighbours = self.findNeighbours(i)
             if(len(neighbours) < self.min_points):
                 self.labels[i] = int(-1)
             else:#if((len(neighbours) >= self.min_points) and  (self.labels[i] == 0 )):
                 self.labels[i] = int(self.cluster_counter)
                 self.completeCluster(i,neighbours)
                 self.cluster_counter += 1
        return self.labels
    def findNeighbours(self,index):
        cluster_neighbours = []
        for j in range(0,self.data.shape[0]):
            if(np.linalg.norm(self.data[index] - self.data[j] ) < self.eps):
                cluster_neighbours.append(j)
        return cluster_neighbours
    def completeCluster(self,index,neighbours):
        k = 0
        while k < len(neighbours):
            new_neighbours = []

            if(self.labels[neighbours[k]] == 0 ):

                self.labels[neighbours[k]] = self.cluster_counter

                new_neighbours = self.findNeighbours(neighbours[k])
                if(len(new_neighbours) >= self.min_points):
                    for m in range(0,len(new_neighbours)):
                        if(new_neighbours[m] not in neighbours):
                            neighbours.append(new_neighbours[m])
            if(self.labels[neighbours[k]] == -1 ):
                self.labels[neighbours[k]] = self.cluster_counter
            k = k + 1
    def sum_of_squarred_error(self,clustered_data,max_cluster):
        for i in range(0 ,max_cluster):
            mean_data = clustered_data[i].mean()
            e = np.linalg.norm(clustered_data[i]-mean_data)
            e = e**2
            e = e.sum()
            #print(e)
            self.error += e
            print(i)
            print(len(clustered_data[i]))

            """
            mean_data = self.clustered_data[i].mean()
            e = self.euclidean_distance(self.clustered_data[i],i)
            e = e**2
            e = e.sum()
            #print(e)
            self.error += e
            """
    def normalize(self,myData):
          myData  = (myData - np.mean(myData))/np.std(myData)
          return myData

loadedData = np.loadtxt("train_3.txt")
x = [row[0] for row in loadedData]
y = [row[1] for row in loadedData]
x = np.array(x)
y = np.array(y)


x = x.reshape(x.shape[0],1)
y = y.reshape(y.shape[0],1)

#data = np.concatenate((x,y), axis = 1)
#data = data[0:1500]
data = np.concatenate((x,y), axis = 1)
orig_data = np.concatenate((x,y), axis = 1)
x = data

std_scale = preprocessing.StandardScaler().fit(x)
x = std_scale.transform(x)

#x = data



neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(x)
distances, indices = nbrs.kneighbors(x)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()


##################
beg = time.time()
f = MyDBSCAN(6,0.26,x)
my_labels = f.fit()
end = time.time()
my_labels = my_labels.T
my_labels = my_labels[0]
my_labels = np.array(my_labels)
my_labels = my_labels.astype(int)
my_labels = my_labels.T

max_labels = my_labels.max()+1
labels_list = []
labels_list.append(-1)
for i in range(0,max_labels+1):
    labels_list.append(i)

labels_list = np.array(labels_list)
my_labels = my_labels.reshape(my_labels.shape[0] , 1)

for i in range(my_labels.shape[0]):
    if(my_labels[i] == -1):
        my_labels[i] = 0
labeled_data = np.concatenate((data,my_labels), axis=1)
max_cluster = np.max(my_labels)
clustred_data_2 = []

print(labels_list.shape)

##how to add minus one to this list????
for i in range(0 ,labels_list.shape[0]):

    tmp = np.where(labeled_data[:,2] == i)
    tmp = np.array(tmp)
    tmp2 = orig_data[tmp, :]
    clustred_data_2.append(tmp2)
    clustred_data_2[i] = np.array(clustred_data_2[i])
    clustred_data_2[i] =clustred_data_2[i].reshape(clustred_data_2[i].shape[1],clustred_data_2[i].shape[2])

clustred_data_2 = np.array(clustred_data_2)
clustred_data = np.array(clustred_data_2)
clustered_data = clustred_data

print(my_labels.T)
for i in labels_list:
    color_c = []

    if(i == 0):
        color_c = ['#000000']
    elif(i == 1):
        color_c = ['#002BFF']
    elif(i == 2):
        color_c = ['#02AB13']
    elif(i == 3):
        color_c = ['#FF0000']
    elif(i == 4):
        color_c = ['#DBC500']
    elif(i == 5):
        color_c = ['#FF00DE']
    elif(i == 6):
        color_c = ['#00FFE6']
    else:
        color_c = randomcolor.RandomColor().generate()
    plt.scatter(clustered_data[i][:,0],clustered_data[i][:,1],color=color_c )

plt.show()
print(len(my_labels.T))


print(" Time: ")
print(end - beg)
f.sum_of_squarred_error(clustred_data,max_cluster)
print(len(clustred_data))
err = f.error
err = np.array(err)
print("Errors : ")
print(err)
