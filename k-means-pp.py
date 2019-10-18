# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:15:16 2018

@author: Kau
"""


import numpy as np
import matplotlib.pyplot as plt
#import math
#import sys

iris = np.loadtxt('iris.data', usecols = (0, 1, 2, 3), delimiter = ',') #load relevant data.

def Euclidian(x, y):
    total = 0
    for i in range(len(x)):
        total += np.square(np.abs(x[i] - y[i]))
    total = np.sqrt(total)
    return total

def sum_of_squares(x, y, total):
    total = 0
    for i in range(len(x)):
        total += np.square(x[i] - y[i])
    return total

def kmeanspp(x_input, K):
    init_centroids = np.copy(x_input[np.random.choice(len(x_input))])
    distance = np.zeros(150) #initialize distance array, re-write if distance = 0.
    for i in range(K):
        for j in range(len(x_input)):
            #get distances between x_input[i] and every closest centroid (thus get distances, save only the lowest one)
            for k in range(len(init_centroids)):
                #get distance
                if i == 0 or i == 1:
                    temp_distance = Euclidian(x_input[j], init_centroids)
                else:
                    temp_distance = Euclidian(x_input[j], init_centroids[k])
                if((temp_distance < distance[j] and temp_distance != 0.0) or distance[j] == 0):
                    distance[j] = temp_distance
        total = np.sum(distance) #get value to divide, to get weights for each distance.
        for j in range(len(distance)):
            distance[j] = distance[j]/ total
#            if(math.isnan(distance[j])): #Check if nan occurs, for bugtesting purposes.
#                sys.exit(1)
        if i == 0:
            init_centroids = np.copy(x_input[np.random.choice(len(x_input), p = distance)])
        else:
            init_centroids = np.vstack((init_centroids, x_input[np.random.choice(len(x_input), p = distance)]))#stack centroids vertically
    return init_centroids
                
    

def k_means_cs171(x_input, k, init_centroids):
    #x_input = matrix of data points.
    #k = number of clusters
    #init_centroids = kxfeature matrix that contains initialization for centroids.
    
    cluster_centroids = init_centroids #start centroids at init values
    cluster_assignments = np.zeros(shape = (len(x_input), 1), dtype = int)
    
    cluster_copy = np.ones(shape = (len(x_input), 1), dtype = int) #use to check if centroids
    #Repeat while cluster assignments don’t change:
    #a) Assign each point to the nearest centroid using Euclidean distance
    #b) Given new assignments, compute new cluster centroids as mean of all points in cluster
    while(not(np.array_equiv(cluster_assignments,cluster_copy))):
#        print(cluster_assignments)
        Num_In_Clust = [[0, 0, 0, 0, 0]]#initialize list to hold totals of each cluster.
        if k > 1:
            for i in range(k-1):
                Num_In_Clust.append([0, 0, 0, 0, 0])#append number of lists required
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
            x_val = np.copy(x_input[i])
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
    x_input = np.copy(iris)

    SSE_total = np.zeros(shape = (10, 1), dtype = float)
    for j in range(100):
        SSE = np.zeros(shape = (10, 1), dtype = float)
        for k in range(1, 11):
            init_centroids = kmeanspp(x_input, k)
            
            [cluster_assignments, cluster_centroids] = k_means_cs171(x_input, k, init_centroids)
            #calculate sum of squares per input.
            for i in range(len(x_input)):
                Send_To = cluster_assignments[i]
                if k != 1:
                    SSE[k-1] += sum_of_squares(x_input[i], cluster_centroids[Send_To][0], SSE[k-1])
                else:
                    SSE[k-1] += sum_of_squares(x_input[i], cluster_centroids, SSE[k-1])
        if j == 0:#exception to initiate SSE_total
            SSE_total = SSE
        else:#append to the right (another column)
            SSE_total = np.hstack((SSE_total, SSE))
        if j == 1 or j == 9 or j == 99:
            SSE_Mean = np.mean(SSE_total, axis = 1, dtype = float)
            SSE_Std_Dev = np.std(SSE_total, axis = 1, dtype = float)
            plt.title('Iris SSE kmeans++ iter = ' + str(j + 1))
            x = np.arange(1, 11)
            plt.errorbar(x, y = SSE_Mean, yerr = SSE_Std_Dev, linestyle='-')
            plt.ylabel('Sum Squared Error')
            plt.xlabel('Num of Clusters')
            plt.savefig('Iris SSE kmeans++ iter = ' + str(j + 1), bbox_inches = 'tight', dpi = 100)
            plt.show()
#        print(x_input)
main()