# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:15:16 2018

@author: Kau
"""


import numpy as np
#import matplotlib.pyplot as plt

iris = np.loadtxt('iris.data', usecols = (0, 1, 2, 3), delimiter = ',')
labels = np.genfromtxt('iris.data', usecols = (4), delimiter = ',', dtype='str')

def Euclidian(x, y):#function for euclidian distance.
    total = 0
    for i in range(len(x)):
        total += np.square(np.abs(x[i] - y[i]))
    total = np.sqrt(total)
    return total

def sum_of_squares(x, y, total): #function for sum of squares distance.
    total = 0
#    print(y)
    for i in range(len(x)):
        total += np.square(x[i] - y[i])
    return total

def k_means_cs171(x_input, k, init_centroids):
    #x_input = matrix of data points.
    #k = number of clusters
    #init_centroids = kxfeature matrix that contains initialization for centroids.
    cluster_assignments = np.zeros(shape = (len(x_input), 1), dtype = int)
    cluster_copy = np.ones(shape = (len(x_input), 1), dtype = int)
    #Repeat while cluster assignments don’t change:
    #a) Assign each point to the nearest centroid using Euclidean distance
    #b) Given new assignments, compute new cluster centroids as mean of all points in cluster
    cluster_centroids = init_centroids #start centroids at init values
    while(not(np.array_equiv(cluster_assignments,cluster_copy))):
        Num_In_Clust = [[0, 0, 0, 0, 0]]
        if k > 1:
            for i in range(k-1):
                Num_In_Clust.append([0, 0, 0, 0, 0])
        cluster_copy = np.copy(cluster_assignments) #set initially to equal, check at the end for equality
        for i in range(len(x_input)):#for each, determine which cluster it is in.
            lowest_val = (1000000.0, -100) #holds distance / which cluster_centroid value / current x_input
            if k > 1:
                for j in range(k):
                    value = Euclidian(x_input[i], init_centroids[j])
                    if value < lowest_val[0]:
                        lowest_val = (value, j)
            else:
                lowest_val = (0, 0) #no point in seeing what centroid to add to if k <= 1.
            #re-calculate centroid
            #Grab cluster for each lowest_val
            index = lowest_val[1]
            x_val = x_input[i]
            cluster_assignments[i] = lowest_val[1]
            for j in range(len(x_val)):
                Num_In_Clust[index][j] = Num_In_Clust[index][j] + x_val[j]
            Num_In_Clust[index][-1] += 1 #Increment by 1 to calculate mean with.
            for j in range(len(x_val)):
                if(cluster_centroids.ndim > 1):
                    cluster_centroids[index][j] = Num_In_Clust[index][j] / Num_In_Clust[index][-1]
                else:
                    cluster_centroids[j] = Num_In_Clust[index][j] / Num_In_Clust[index][-1]
    return cluster_assignments, cluster_centroids
    
def main():
    #test k = 3
    x_input = iris
    init_centroids = iris[1]
    for j in range(1):#for one iteration 
        for k in range(3, 4):#of k = 3
            #initialize random values, populate init_centroids, run algorithm.
            row_len, col_len = x_input.shape #grab row length, for use in initialization of random variables.
            rand_vals = np.random.choice(row_len, k, replace = False)
            init_centroids = iris[rand_vals, :]
            [cluster_assignments, cluster_centroids] = k_means_cs171(x_input, k, init_centroids) #run k-means.
            in_cluster = [[], [], []] #list of lists, will hold indexes that are in each cluster, 0 = cluster 0, so on.
            for i in range(len(cluster_assignments)):
                in_cluster[cluster_assignments[i][0]].append(i) # Hold all the values 
            dist = [[], [], []]# distances, holds distances of each value in the cluster, to the cluster.
            #Get distances
            for i in range(len(in_cluster)):
                for j in range(len(in_cluster[i])):
                    dist[i].append(Euclidian(cluster_centroids[i], x_input[in_cluster[i][j]])) 
            temp = [[], [], []]
            #get indices for distances for each cluster.
            for i in range(3):
                for j in range(3):
                    temp[i].append((min(dist[i]), dist[i].index(min(dist[i]))))
                    del dist[i][temp[i][-1][1]]
            result = []
            #get result
            for i in range(3):
                for j in range(3):
                    result.append(labels[temp[i][j][1]]) #grab labels
            temp2 = [0, 0, 0]#check if any of the values are Iris-setosa.
            temp2[0] = result[0:3].count('Iris-setosa')
            temp2[1] = result[3:6].count('Iris-setosa')
            temp2[2] = result[6:9].count('Iris-setosa')
            for i in range(3):
                if temp2[i] >= 2: #if value is < 2, then the other label is matching.
                    print('There are ' + str(temp2[i]) + ' matching labels in cluster ' + str(i))
                else:
                    print('There are ' + str(3 - temp2[i]) + ' matching labels in cluster ' + str(i))
main()