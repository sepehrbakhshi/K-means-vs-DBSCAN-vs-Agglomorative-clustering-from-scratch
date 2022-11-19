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
import time
import randomcolor
from sklearn import preprocessing
class HAC(object):

    def __init__(self, k , data ,c):

       self.number_of_clusters = k
       self.dimensions  = 2
       self.data = data
       self.labels =  np.zeros(shape=(self.data.shape[0],1))
       self.children = []
       self.node  = np.zeros(shape=(self.data.shape[0],1))
       self.number_of_leaves = []
       self.final_nodes = []
       self.node_data = []
       self.error = 0
       self.clustering_method =c
     #calculating euclidean distance
    def euclidean_distance(self,mydata):
        eu_pow = 0
        eu_distance = 0
        eu_sum = 0
        #x = mydata[[row[i] for row in mydata] for i in range(0,5)]
        eu_pow = self.data - mydata
        eu_pow = eu_pow**2
        eu_pow = np.array(eu_pow)
        eu_sum = eu_pow.sum(axis=1)
        eu_distance = np.sqrt(eu_sum)
        return eu_distance

    def euclidean_distance2(self,mydata,center):

        eu_pow = 0
        eu_distance = 0
        eu_sum = 0
        eu_pow = (mydata - center)
        #print(eu_pow[0])
        eu_pow = eu_pow**2
        eu_pow = np.array(eu_pow)
        eu_sum = eu_pow.sum(axis=1)
        eu_distance = np.sqrt(eu_sum)
        return eu_distance
    #doing the hiearchial clustering
    def cluster(self):
        counter = 0
        distance_matrix = [self.euclidean_distance(i) for i in self.data]
        distance_matrix = np.array(distance_matrix)
        np.fill_diagonal(distance_matrix, float('Inf'))

        distance_matrix = np.array(distance_matrix)

        for i in range(0,self.data.shape[0]-1):
            x = np.min(distance_matrix)



            min_location = np.where(distance_matrix == x)
            min_location = np.array(min_location)
            tmp_min_loc = []
            if(min_location.shape[1] > 2):
                #min_location = np.delete(min_location,[1,2,3],1)
                min_location = min_location[:,0]
                min_location = min_location.reshape(2,1)
                rev_1 = min_location[1,0]
                rev_2 = min_location[0,0]
                tmp_min_loc.append(rev_1)
                tmp_min_loc.append(rev_2)
                tmp_min_loc = np.array(tmp_min_loc)
                tmp_min_loc = tmp_min_loc.reshape(2,1)
                min_location = np.concatenate((min_location,tmp_min_loc), axis = 1)


            #print(min_location)

            if(self.node[min_location[0,0]] == 0 and self.node[min_location[0,1]] == 0):
                self.children.append(np.array([min_location[0,0] , min_location[0,1]]))
                self.number_of_leaves.append(2)
                self.node_data.append(np.array([min_location[0,0] , min_location[0,1]]))
                #print(self.children)
            elif(self.node[min_location[0,0]] == 0  and self.node[min_location[0,1]] != 0 ):

                a =int( min_location[0,0]  )
                b =int( self.node[min_location[0,1]])

                a_ = min(a,b)
                b_ = max(a,b)
                child = []
                child.append(a_)
                child.append(b_)
                child = np.array(child)
                self.children.append(child)
                self.number_of_leaves.append((1 + self.number_of_leaves[b-self.data.shape[0]]))

                first = min_location[0,0]
                second = self.node_data[b_-self.data.shape[0]]
                this_node = np.hstack((first,second))
                self.node_data.append(this_node)

            elif(self.node[min_location[0,0]] != 0  and self.node[min_location[0,1]] == 0 ):

                a =int( self.node[min_location[0,0]] )
                b =int( min_location[0,1] )
                a_ = min(a,b)
                b_ = max(a,b)
                child = []
                child.append(a_)
                child.append(b_)
                child = np.array(child)
                self.children.append(child)

                self.number_of_leaves.append((1 + self.number_of_leaves[a-self.data.shape[0]]))

                first = self.node_data[b_ - self.data.shape[0]]
                second =  min_location[0,1]
                this_node = np.hstack((first,second))
                self.node_data.append(this_node)

            else:
                a =int( self.node[min_location[0,0]])
                b =int( self.node[min_location[0,1]])
                a_ = min(a,b)
                b_ = max(a,b)
                child = []
                child.append(a_)
                child.append(b_)
                child = np.array(child)
                self.children.append(child)
                self.number_of_leaves.append((self.number_of_leaves[b-self.data.shape[0]] + self.number_of_leaves[a-self.data.shape[0]]))

                first = self.node_data[a_ - self.data.shape[0]]
                second =  self.node_data[b_ - self.data.shape[0]]
                this_node = np.hstack((first,second))
                #this_node = this_node.reshape(1,this_node.shape[1])

                self.node_data.append(this_node)


            self.node[min_location[0,0]] = self.data.shape[0] + counter
            self.node[min_location[0,1]] = self.data.shape[0] + counter

            two_rows = []
            two_rows = (distance_matrix[min_location[0,0:2]])


            #average Linkage
            print(self.clustering_method)
            if( self.clustering_method  == 'average'):
                two_rows = np.sum(two_rows,axis = 0)
                print("yes")
                two_rows = two_rows/2
            elif( self.clustering_method  == 'single'):

                two_rows = two_rows.min(axis = 0 )
            elif(self.clustering_method  == 'complete'):
                two_rows = two_rows.max(axis = 0 )

            two_rows = np.array(two_rows)


            distance_matrix[min_location[0,1]] = two_rows
            distance_matrix[:,min_location[0,1]] = two_rows

            distance_matrix[min_location[0,0]] = float('Inf')
            distance_matrix[:,min_location[0,0]] = float('Inf')
            np.fill_diagonal(distance_matrix, float('Inf'))

            counter = counter +1

        self.children = np.array(self.children)

        root = self.children[-1]
        root = np.array(root)
        self.final_nodes = np.array(self.final_nodes)
        limit = self.number_of_clusters

        if(self.number_of_clusters == 1):
                self.final_nodes = np.hstack([self.final_nodes, [self.data.shape[0] + self.children.shape[0]]])
        elif(self.number_of_clusters == 2 ):
                self.final_nodes = np.hstack([self.final_nodes, root[0]])
                self.final_nodes = np.hstack([self.final_nodes, root[1]])
        if(self.number_of_clusters == self.data.shape[0]):
                root = [0] * self.data.shape[0]

        else:
            while(root.shape[0] < limit):
                node = []
                for i in root:
                    node = np.max(root)
                i = np.where(root == np.max(node))
                i = int(i[0])
                root[i] = self.children[(np.max(node) - self.data.shape[0])][0]
                root = np.insert(root , i + 1 ,self.children[(np.max(node) - self.data.shape[0])][1])

        root = np.array(root)
        self.final_nodes = root
        self.node_data = np.array(self.node_data)

        self.node_data = self.node_data.reshape(self.node_data.shape[0],1)

        label = int(0)
        #self.final_nodes = np.sort(self.final_nodes)

        #self.final_nodes = self.final_nodes[::-1]

        for node in self.final_nodes:

            if(self.number_of_clusters == self.data.shape[0]):
                self.labels = np.arange(0,self.data.shape[0])
                break
            elif(node > self.data.shape[0]-1 ):
                n = node - self.data.shape[0]
                cluster = self.node_data[n]
                self.labels[cluster[0]] = label
            else:
                cluster = node
                cluster = np.array(cluster)
                self.labels[cluster] = label

            label +=1

        self.labels = np.array(self.labels)
        self.labels = self.labels.astype(int)
        self.labels = self.labels.reshape(1 ,self.labels.shape[0])
    #calculating the error
    """
    def sum_of_squarred_error(self,my_data,centers):
        for i in range(0 ,self.number_of_clusters):
            e = np.linalg.norm(my_data[i]-centers[i])
            e = self.euclidean_distance2(my_data[i],centers[i])
            e = e**2
            e = e.sum()
            self.error += e
    """
    def sum_of_squarred_error(self,clustered_data):
        for i in range(0 ,self.number_of_clusters):
            mean_data = clustered_data[i].mean()
            e = np.linalg.norm(clustered_data[i]-mean_data)
            e = e**2
            e = e.sum()
            #print(e)
            self.error += e
            print(i)
            """
            mean_data = self.clustered_data[i].mean()
            e = self.euclidean_distance(self.clustered_data[i],i)
            e = e**2
            e = e.sum()
            #print(e)
            self.error += e
            """
###############################################################################
#######################MAIN CLASS STARTS#######################################
"""
instead of image ,I read the labels and centered created by the K means
for 2 hundred clusters
"""
"""
labels = np.loadtxt("preprocessed_labels.txt")
centers = np.loadtxt("preprocessed_centers.txt")
labels = labels.reshape(labels.shape[0],1)

x = centers
x = np.array(x)
"""
loadedData = np.loadtxt("train_3.txt")
x = [row[0] for row in loadedData]
y = [row[1] for row in loadedData]
x = np.array(x)
y = np.array(y)

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
#data = data[0:200]
x = data
"""
from sklearn.preprocessing import normalize
x = normalize(x, axis=0, norm="max")
"""
std_scale = preprocessing.StandardScaler().fit(x)
x = std_scale.transform(x)



#print( model.children_[:20])

beg = time.time()
"""
You can define the clustering method here:
'single' for single Linkage
'complete' for complete Linkage
'average' for average Linkage
"""
f = HAC(2,x,'single')
f.cluster()
end = time.time()
print("My Clustering")
np.set_printoptions(threshold=sys.maxsize)
counter = 0

#print(f.labels.T.shape)
labeled_data = np.append(orig_data, f.labels.T, axis=1)

clustred_data_2 = []
for i in range(0 ,f.number_of_clusters):

    tmp = np.where(labeled_data[:,2] == i)

    tmp = np.array(tmp)
    tmp2 = orig_data[tmp, :]
    clustred_data_2.append(tmp2)
    clustred_data_2[i] = np.array(clustred_data_2[i])
    clustred_data_2[i] =clustred_data_2[i].reshape(clustred_data_2[i].shape[1],clustred_data_2[i].shape[2])
clustred_data_2 = np.array(clustred_data_2)
clustred_data = np.array(clustred_data_2)


centers = np.zeros(shape=(f.number_of_clusters ,f.dimensions))
i = 0
for c in clustred_data:
    v = c.sum(axis=0)
    centers[i] = v/(c.shape[0]+0.00000001)
    i = i + 1
print("Centers: ")
print(centers)
print("******************")
#centers
f.sum_of_squarred_error(clustred_data)
print("ERROR :")
print(f.error)

print("Time: " )
print(end- beg)


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
        #print(randomcolor.RandomColor().generate())
    plt.scatter(clustred_data[i][:,0],clustred_data[i][:,1],color=color_c)
plt.show()
