# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:15:16 2018

@author: Kau
"""


import numpy as np
import matplotlib.pyplot as plt

iris = np.loadtxt('iris.data', usecols = (0, 1, 2, 3), delimiter = ',')

def Euclidian(x, y):
    total = 0
    for i in range(len(x)):
        total += np.square(np.abs(x[i] - y[i]))
    total = np.sqrt(total)
    return total

def sum_of_squares(x, y, total):
    total = 0
#    print(y)
    for i in range(len(x)):
        total += np.square(x[i] - y[i])
    return total

def k_means_cs171(x_input, k, init_centroids):
    #x_input = matrix of data points.
    #k = number of clusters
    #init_centroids = kxfeature matrix that contains initialization for centroids.
    #Repeat while cluster assignments don’t change:
    #a) Assign each point to the nearest centroid using Euclidean distance
    #b) Given new assignments, compute new cluster centroids as mean of all points in cluster
    cluster_centroids = init_centroids #start centroids at init values
    cluster_assignments = np.zeros(shape = (len(x_input), 1), dtype = int)
    
    cluster_copy = np.ones(shape = (len(x_input), 1), dtype = int)
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
#            print(lowest_val[1])
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
    
    SSE_total = np.zeros(shape = (10, 1), dtype = float)
    for j in range(100):
        SSE = np.zeros(shape = (10, 1), dtype = float)
        for k in range(1, 11):
            #initialize random values, populate init_centroids, run algorithm.
            row_len, col_len = x_input.shape
            rand_vals = np.random.choice(row_len, k, replace = False)
            init_centroids = iris[rand_vals, :]
            [cluster_assignments, cluster_centroids] = k_means_cs171(x_input, k, init_centroids)
            #calculate sum of squares per input.
            for i in range(len(x_input)):
                Send_To = cluster_assignments[i]
                SSE[k-1] += sum_of_squares(x_input[i], cluster_centroids[Send_To][0], SSE[k-1])
        if j == 0:#exception to initiate SSE_total
            SSE_total = SSE
        else:#append to the right (another column)
            SSE_total = np.hstack((SSE_total, SSE))
        if j == 1 or j == 9 or j == 99: #only three plots needed
            SSE_Mean = np.mean(SSE_total, axis = 1, dtype = float)
            SSE_Std_Dev = np.std(SSE_total, axis = 1, dtype = float)
            plt.title('Iris SSE iter = ' + str(j + 1))
            x = np.arange(1, 11)
            plt.errorbar(x, y = SSE_Mean, yerr = SSE_Std_Dev, linestyle='-')
            plt.ylabel('Sum Squared Error')
            plt.xlabel('Num of Clusters')
            plt.savefig('Iris SSE kmeans iter = ' + str(j + 1), bbox_inches = 'tight', dpi = 100)
            plt.show()
main()