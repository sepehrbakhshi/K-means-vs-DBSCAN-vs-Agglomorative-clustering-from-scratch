# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:22:25 2019

@author: Sepehr
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import randomcolor
from sklearn.cluster import KMeans
import time
from sklearn import preprocessing
class k_means(object):

    #initilize with some values
    def __init__(self, k , data):

       self.number_of_clusters = k
       self.dimensions  = 2
       self.centers = np.zeros(shape=(self.number_of_clusters ,self.dimensions))

       self.data = data
       self.labels = []
       self.error = 0
       self.clustered_data = []
       #picking some values randomly from image as starting point
       for i in range(0,self.number_of_clusters):
           index = np.random.rand()*self.data.shape[0]
           index = int(index)
           print(index)
           print(self.data[index])
           self.centers[i] = self.data[index]


    #calculating distance
    def euclidean_distance(self,mydata,cluster):
        eu_pow = 0
        eu_distance = 0
        eu_sum = 0
        eu_pow = (mydata - self.centers[cluster])

        eu_pow = eu_pow**2
        eu_pow = np.array(eu_pow)
        eu_sum = eu_pow.sum(axis=1)
        eu_distance = np.sqrt(eu_sum)
        return eu_distance
    #clustering the data
    def cluster(self):
        distances = []
        for i in range(0, self.number_of_clusters):
            distances.append(self.euclidean_distance(self.data,i))
        distances = np.array(distances)

        self.labels = np.argmin(distances,axis=0)
        self.labels = self.labels.reshape(self.labels.shape[0],1)
        labeled_data = np.append(self.data, self.labels, axis=1)
        #print(self.labels[0:100])
        clustred_data_2 = []
        for i in range(0 ,self.number_of_clusters):

            tmp = np.where(labeled_data[:,2] == i)

            tmp = np.array(tmp)
            tmp2 = self.data[tmp, :]
            clustred_data_2.append(tmp2)
            clustred_data_2[i] = np.array(clustred_data_2[i])
            clustred_data_2[i] =clustred_data_2[i].reshape(clustred_data_2[i].shape[1],clustred_data_2[i].shape[2])
        clustred_data_2 = np.array(clustred_data_2)
        clustred_data = np.array(clustred_data_2)
        self.clustered_data = clustred_data

        i = 0
        for c in clustred_data:
            v = c.sum(axis=0)
            self.centers[i] = v/(c.shape[0]+0.00000001)
            i = i + 1
    #here as i mentioned in the report I use 10 as my number of iteration and threshold
    def find_best_cluster(self):
        threshold = 19
        for i in range(0,threshold):
            self.cluster()
    #calculating error
    def sum_of_squarred_error(self,clustered_data):
        for i in range(0 ,self.number_of_clusters):
            mean_data = clustered_data[i].mean()
            print(mean_data)
            e = np.linalg.norm(clustered_data[i]-mean_data)
            e = e**2
            e = e.sum()
            #print(e)
            self.error += e

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
###############################################################################
#######################MAIN CLASS STARTS#######################################
#Read the image
"""
img = cv2.imread('sample.jpg')
x = img.reshape((-1,3))
"""
loadedData = np.loadtxt("train_3.txt")
x = [row[0] for row in loadedData]
y = [row[1] for row in loadedData]
x = np.array(x)
y = np.array(y)
print(np.max(x))
print(np.max(y))
print(x.shape)
print(y.shape)
print(x[0])
print(y[0])
x = x.reshape(x.shape[0],1)
y = y.reshape(y.shape[0],1)
print(x.shape)
print(y.shape)
data = np.concatenate((x,y), axis = 1)
orig_data = np.concatenate((x,y), axis = 1)
#data = data[0:10]
x = data
"""
from sklearn.preprocessing import normalize
x = normalize(x, axis=0, norm="l2")
"""
std_scale = preprocessing.StandardScaler().fit(x)
x = std_scale.transform(x)

errors = []

x = np.float32(x)
"""
Very Important:
    Here we can change the number of clusters
    below the initial number of clusters are 2
"""
beg = time.time()

clustering = k_means(2, x)
#x = clustering.normalize(x)
clustering.find_best_cluster()
clustred_data_2 = []
labeled_data = np.concatenate((orig_data,clustering.labels), axis = 1)
for i in range(0 ,clustering.number_of_clusters):

    tmp = np.where(labeled_data[:,2] == i)

    tmp = np.array(tmp)
    tmp2 = orig_data[tmp, :]
    clustred_data_2.append(tmp2)
    clustred_data_2[i] = np.array(clustred_data_2[i])
    clustred_data_2[i] =clustred_data_2[i].reshape(clustred_data_2[i].shape[1],clustred_data_2[i].shape[2])
clustred_data_2 = np.array(clustred_data_2)
clustred_data = np.array(clustred_data_2)

clustering.sum_of_squarred_error(clustred_data)
end = time.time()
print("------")
print("time : ")
print(end - beg)
print("------")
#errors.append(clustering.error)
"""
clustering.centers = np.uint8(clustering.centers)

res = clustering.centers[clustering.labels.flatten()]
"""
#res2 = res.reshape((img.shape))

numbers = np.zeros(shape=(3,1))
#path = 'C:/Users/Sepehr/Documents'
print("Centers :")
print(clustering.centers)
print("Error :")
print(clustering.error)
#plt.scatter(orig_data[:,0], orig_data[:,1])


#labeled_data = np.append(orig_data, self.labels, axis=1)

for i in range(0, 2):
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
    plt.scatter(clustred_data[i][:,0],clustred_data[i][:,1],color=color_c)
plt.show()



k = []
"""
for i in range(1,7):
    clustering = k_means(i, x)
    clustering.find_best_cluster()
    clustering.sum_of_squarred_error(clustred_data)
    errors.append(clustering.error)
    k.append(i)
"""
print("------")
print("time : ")
print(end - beg)
print("------")
#errors = np.array(errors)
#print(errors.sum())
#plt.plot(k, errors)
plt.show()

print("error")
