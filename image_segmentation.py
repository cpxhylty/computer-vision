# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 20:15:52 2020

@author: ysxh1998
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm#千万不要写成import tqdm,不然就会有'module' object is not callable
from copy import deepcopy


imageA = plt.imread(r'shan.jpg')#cv2,plt读出来的直接是numpy
type(imageA)#numpy
imageA.shape
plt.imshow(imageA)
instances = np.zeros((imageA.shape[0]*imageA.shape[1],3))
instances.shape
image_normalized = imageA/255
feature1 = np.squeeze(image_normalized[:,:,0].reshape(120000,-1))
feature2 = np.squeeze(image_normalized[:,:,1].reshape(120000,-1))
feature3 = np.squeeze(image_normalized[:,:,2].reshape(120000,-1))

instances[:,0] = feature1
instances[:,1] = feature2
instances[:,2] = feature3



class KMeans_seg(object):
    """
    KMeans Clustering Algorithm, an unsupervised learning algorithm.
        Attributes: 
        ===========
            'num_clusters_': (K) num of clusters taken as input.
            'centroids_': Initially, the centroids taken as input, 
            but contains the final centroids produced by algorithm after calling fit(X).
            'num_restarts_': The number of restarts taken as input.
            'max_iter_': The maximum number of iterations taken as input.
            'tolerance_': The tolerance value taken as input.
            'iter_num_': The number of iterations the algorithm have reached to converge.
            'distoration_measure_': The distorartion measure of the final assignment of algorithm.
            'cluster_labels_': The Clusters Labels vector to indicate each instance's cluster.
    """

    def __init__(self, num_clusters = 2, init_seed = None, num_restarts = 0, max_iter = 300, tolerance = 0.0001):
        """
        Parameters: 
        ----------
            'num_clusters' : int, default: 2
                The number of clusters (K) to form as well as the number of centroids to generate.
            'init_seed' : None or an ndarry, default: None
                If None will generate k random initial centroids.
                If ndarry initial centroids is passed, it should be of shape (num of clusters, num of features).
            'num_restarts' : int, default: 0
                Number of times (N) the K-means algorithm will be run with different centroids.
                The final results will be the best output of the (N) runs.
            'max_iter' : int. default: 300
                Maximum number of iterations of the k-means algorithm for a single run.
            'tolerance' : float, default: 0.0001
                Relative tolerance with regards to inertia to declare convergence.
        """
        super().__init__()

        self.num_clusters_ = num_clusters
        self.centroids_ = init_seed
        self.num_restarts_ = num_restarts
        self.max_iter_ = max_iter
        self.tolerance_ = tolerance
        self.iter_num_ = 0
        self.distoration_measure_ = 0
        self.cluster_labels_ = np.array([])
        self._is_centroids_random = init_seed is None
        self._measure_history = []

    def fit(self, data): 
        """
        Run KMeans algorithm on some data.
        Parameters:
        -----------
            'data': n-dimensional numpy array that represents the data to be clustered.
        """
        num_features = data.shape[1]
        num_instances = data.shape[0]
        best_restart_score = np.inf
        for restart_num in tqdm(range(self.num_restarts_+1)):
            # If initial seed is None, then we have to generate the centroids randomly from data.
            if(self._is_centroids_random):
                self.centroids_ = self._random_initialize_centroids(data, num_features)
            # Old centers, to store old centers from last iteration. Initally Zeros.      
            old_centroids = np.zeros(self.centroids_.shape) 
            # New centers, to store updated centroids each iteration.
            new_centroids = deepcopy(self.centroids_) 
            # Error, to compute error each iteration of KMeans for convergence.
            error = np.linalg.norm(new_centroids - old_centroids)
            # Distaces, to store calculations of distances between all points and centroids.
            distances = np.zeros((num_instances,self.num_clusters_))
            # Distoration measure, The 'quality' of the current assignment is given by it.
            distoration_measure = 0
            # Cluster Labels, to store a vector of cluster labels for each instance 
            cluster_labels = np.zeros(num_instances)
            # Iteration Params
            progress_bar = tqdm(total=self.max_iter_)
            iter_num = 0
            measure_history = []

            # When, after an update, the estimate of that center stays the same, exit loop
            while error > self.tolerance_ and iter_num < self.max_iter_:
                # Measure the distance to every center
                for i in range(self.num_clusters_):
                    distances[:,i] = np.linalg.norm(data - new_centroids[i], axis=1)

                # Assign all training data to closest center
                cluster_labels = np.argmin(distances, axis = 1)
                old_centroids = deepcopy(new_centroids)
                # Calculate mean for every cluster and update the center
                for i in range(self.num_clusters_):
                    new_centroids[i] = np.mean(data[cluster_labels == i], axis=0)
                error = np.linalg.norm(new_centroids - old_centroids)
                distoration_measure = np.sum(distances)
                measure_history.append(distoration_measure)
                print("Restart",restart_num+1," Iteration", iter_num+1)
                print("The Error of this iteration is ", error)
                print("The Distoration Measure score of this assignment is ", distoration_measure)
                iter_num +=1
                progress_bar.update(1)

            progress_bar.close()
            if(distoration_measure < best_restart_score):
                print("This Restart scored better than last _plorone. Updating Attributes...")
                self.cluster_labels_ = cluster_labels
                self.distoration_measure_ = distoration_measure
                self._measure_history = measure_history
                self.iter_num_ = iter_num
            else: 
                print("This Restart have lower score than best one. Ignoring...")
        
    def _random_initialize_centroids(self, data, num_features):
        """Private Method
        Generate random centers. Uses standard deviation and mean to ensure it represents the whole data."""
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        return np.random.randn(self.num_clusters_,num_features)*std + mean

    def plot_measure_vs_iteration(self):
        plt.figure(figsize=(15, 10))
        plt.plot(range(self.iter_num_), self._measure_history)
        

num = 4
num_clusters=num
model = KMeans_seg(num_clusters)
model.fit(instances)

for i,centroid in enumerate(model.centroids_):
    centroid = (centroid*255).astype(int)
    print(centroid)

m = model.cluster_labels_
m.shape
for i in range(num_clusters):
    print(np.sum(m==i))

classid = []
for i in range(num_clusters):
    temp = np.where(m==i)[0]
    classid.append(temp)

feature1 = np.zeros(120000)
feature2 = np.zeros(120000)
feature3 = np.zeros(120000)

centroids = (model.centroids_*255).astype(np.uint8)
centroids[1][1] -= 5000
color = np.array([[51,149,247],[181,119,125],[192,192,192],[10,112,28]]).astype(int)
for i,member in enumerate(classid):
    feature1[member] = centroids[i][0]
    feature2[member] = centroids[i][1]
    feature3[member] = centroids[i][2]
    
    
layer1 = feature1.reshape(300,400)
layer2 = feature2.reshape(300,400)
layer3 = feature3.reshape(300,400)  

new_picture = np.dstack([layer1,layer2,layer3])

plt.imshow(new_picture)
print('!!!')
print(new_picture[0][0][1:10])
plt.imsave(r'shan2.jpg',new_picture)

centroids
















