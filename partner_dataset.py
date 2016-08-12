#########################################################
#       Looking at Missing Data in the Partner Dataset  #
#########################################################

import os
import math
os.chdir("/Users/spencerhall/Desktop/UdacityCode/speed_dating")
from speed_dating_preprocessing import partner_dataset as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


finding_missing = p.notnull()
missing_per_row = []
for i in p.axes[0]:
    missing_per_row.append(np.mean(finding_missing.loc[i]))

# Using a histogram, we can visualize what percentage of data each observation
# in this dataset has.
# https://bespokeblog.wordpress.com/2011/07/11/basic-data-plotting-with-matplotlib-part-3-histograms/
plt.hist(missing_per_row, bins=50)
plt.show()

# It looks like the vast majority of them have over 90% of the data
# for each column. Let's see what percentage have at least 90% of the data.

def AtLeast90(arg):
    return arg >= 0.90
    
print np.mean(map(AtLeast90, missing_per_row))

# About 92.52%, or 7750 out of the total 8378 observations have at least
# 90% of the data. Data imputation seems safe to use on these elements because
# of the paucity of missing values; the observations which have more than 10%
# of their data missing can be discarded, leaving us with close to 7800
# data points.

# The next thing we should do is to find out whether certain variables
# are less likely to have observed values.

columns = p.axes[1]
missing_per_col = []
for i in columns:
    missing_per_col.append(np.mean(finding_missing[i]))
    
plt.hist(missing_per_col, bins=50)
plt.show()

# It appears that the great majority of the variables have over 95% of their
# values in each observation. Let's check to see what percentage have at least
# 95% of their values per observation, and then get the names of those that
# don't.

def AtLeast95(arg):
    return arg >= 0.95

at_least_95 = map(AtLeast95, missing_per_col)
print np.mean(at_least_95) # ~85.7%
below_95 = []
for i in range(0, len(at_least_95)):
    if at_least_95[i] == False:
        below_95.append(i)
        
print list(p.axes[1][below_95])
# print missing_per_col[below_95]

for i in below_95:
    print "\nVariable:", p.axes[1][i], "\nPercentage present:", missing_per_col[i]
    

# amb, shar, amb_o, and shar_o represent two partners' ratings of each other
# on ambitiousness and shared interests. These seem like potentially important
# variables to explore, and since they are present for about 90% of observations,
# it seems reasonable to keep these variables and use multiple imputation for
# their missing values. positin1 is the station number where the individual
# started at the event; it isn't relevant for our study here, which is about
# broader principles that influence romantic matching, and the same with zipcode.
# income, tuition, expnum (expected number of people who will like you), and
# mn_sat (median SAT score) all seem interesting, but have far too few 
# observations to be useful.

# To conclude the selection of variables, I will retain all of the variables
# which are present for 95% or more of the observations, as well as 
# amb, shar, amb_o, and shar_o.

to_keep = []
for i in range(0, len(at_least_95)):
    if at_least_95[i] == True:
        to_keep.append(columns[i])
        
to_keep += ["amb_o", "shar_o", "amb", "shar"]

# Finally, before doing data imputation, we check to see what percentage
# of observations now have at least 95% of their variables now that 
# the variables with many missing values have been removed.

p_new = p[to_keep]
finding_missing = p_new.notnull()
missing_per_row = []
for i in p.axes[0]:
    missing_per_row.append(np.mean(finding_missing.loc[i]))  
print np.mean(map(AtLeast90, missing_per_row))

# With the frequently-missing variables removed, about 95.4% of the observations
# have over 90% of their variables observed.

#########################################################
#         Data Imputation in the Partner Dataset        #
#########################################################

# My task now is to use data imputation to fill in the missing values
# in this dataset. I will use the mean of each column as the stand-in
# value for each missing data point in a given column.


for i in p_new.axes[1]:
    if type(p_new[i][0]) == np.int64 or type(p_new[i][0]) == np.float64:
        mu = np.nanmean(p_new[i])
        if np.mean(p_new[i].isnull()) > 0:
            print i, "has missing values"
            p_new[i].fillna(mu, inplace=True)
            print p_new[i].isnull()
    else:
        print i


p_new = p_new.drop(["from", "career", "field"], axis=1)


#########################################################
#    Principal Component Analysis in Three Dimensions   #
#########################################################

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from math import log

def Log_Series(S):
    log_S = []
    for i in S:
        if float(i) > 0:
            log_S.append(log(float(i)))
        else:
            log_S.append(log(0.0000001))
    return(pd.Series(log_S))

logp_new = p_new.apply(Log_Series, axis=1)


pca = PCA(n_components = 3)
t = pca.fit(logp_new)
transform = pca.fit_transform(logp_new)


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=38, azim=130)
plt.cla()
ax.scatter(transform[:, 0], transform[:, 1], transform[:, 2], 
           cmap=plt.cm.spectral)
plt.show()


#########################################################
#           K-MEANS CLUSTERING IN THREE DIMENSIONS      #
#########################################################

###### This block of code uses an elbow plot to find the optimal number
###### cluster centroids for K-Means clustering.
try_centroids = range(2, 16)
inertia = []
for i in try_centroids:
    cluster = KMeans(n_clusters=i, n_init=30)
    c_pca = cluster.fit(transform)
    inertia.append(c_pca.inertia_)
    
# Elbow plot
plt.plot(try_centroids, inertia, "ro")
plt.show()

####### Now we use the optimal number, 8, to perform and graph the clustering
####### three dimensions.

cluster = KMeans(n_clusters=8, n_init=30)
c_pca = cluster.fit(transform)

fig = plt.figure(1, figsize=(5, 5))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=38, azim=130)
plt.cla()
ax.scatter(transform[:, 0], transform[:, 1], transform[:, 2], 
           c=c_pca.labels_, cmap=plt.cm.spectral)
plt.show()
    
##############################################################
#  Finding Optimal Parameters for PCA and K-Means Clustering #
##############################################################
    
try_n = range(2, 50)
explained_variance = []
for i in try_n:
    learn = PCA(n_components=i)
    t = learn.fit(logp_new)
    explained_variance.append(sum(t.explained_variance_ratio_))
    
plt.plot(try_n, explained_variance, "ro")
plt.show()
    
# The ideal number appears to be 7. Now let's fit the PCA model with 7
# components.
    
final_fit = PCA(n_components=7)
t = final_fit.fit_transform(logp_new)

# Then use the elbow plot to find the ideal number of clusters for the points
# in these seven dimensions.

try_centroids_2 = range(2, 70)
inertia_2 = []
for i in try_centroids_2:
    cluster = KMeans(n_clusters=i, n_init=30)
    c_pca = cluster.fit(t)
    inertia_2.append(c_pca.inertia_)
    
# Elbow plot again
plt.plot(try_centroids_2, inertia_2, "ro")
plt.show()

# Perform K-Means clustering with 10 clusters, which appears to be optimal,
# and then get the cluster centers.

high_d = KMeans(n_clusters=10, n_init=30)
high_d_cluster = high_d.fit(t)
centers = high_d_cluster.cluster_centers_
back_to_log = final_fit.inverse_transform(centers)
real_data_centers = np.round(np.exp(back_to_log))

#################### To do now:
# 1. Interpret the cluster centers from the 7-dimensional PCA.
# 2. Do the PCA / clustering again with a few different subsets of the data,
# and interpret them too.
# 3. Come up with some final ML-related thing and use it.

#########################################################
#          Compressing Dataset by Individuals           #
#########################################################

to_compress = list(np.arange(1.0, 23.0))
compress_df = pd.DataFrame()
for j in to_compress:
   blah = np.array([p_new["id"]])
   indices = list((np.where(blah == j)[1]))
   df_indices = []
   for i in range(0, len(indices)):
       if i != (len(indices) - 1):
           if (indices[i + 1] - indices[i] <= 1):
               df_indices.append(indices[i])
           elif (indices[i + 1] - indices[i] > 1):
               nelson = (p_new.loc[df_indices]).apply(func=np.mean, axis=0)
               goodman = (pd.DataFrame(nelson)).transpose()
               compress_df = pd.concat([compress_df, goodman])
               df_indices = []
       else:
           nelson = (p_new.loc[df_indices]).apply(func=np.mean, axis=0)
           goodman = (pd.DataFrame(nelson)).transpose()
           compress_df = pd.concat([compress_df, goodman])
           df_indices = []
           # End of loops
           
#########################################################
#    Final Cleaning and Exploratory Visualization       #
#########################################################

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from math import log

def Log_Series(S):
    log_S = []
    for i in S:
        if float(i) > 0:
            log_S.append(log(float(i)))
        else:
            log_S.append(log(0.0000001))
    return(pd.Series(log_S))

logp_new = compress_df.apply(Log_Series, axis=1)
# http://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
logp_new.columns = [compress_df.axes[1]]
logp_new = logp_new.drop(["id", "idg", "condtn", "wave", "round", "position",
                         "order", "partner", "pid", "int_corr"], axis=1)
                         
import matplotlib.pyplot as plt
import scipy
import pylab
means = [np.mean(compress_df["attr_o"]), np.mean(compress_df["sinc_o"]), 
     np.mean(compress_df["fun_o"]), np.mean(compress_df["intel_o"])]

# Citation for how to make a barplot:
# http://stackoverflow.com/questions/2177504/individually-labeled-bars-for-bar-graphs-in-matplotlib-python
x = scipy.arange(4)
y = scipy.array(means)
fig = pylab.figure()
ax = fig.add_axes([0.8, 0.8, 0.8, 0.8])
ax.bar(x, y, align="center")
ax.set_xticks(x)
ax.set_xticklabels(["attr_o", "sinc_o", "fun_o", "intel_o"])
fig.show()


######################################################
#     PCA and K-Means Clustering for Each Subgroup   #
######################################################

# This is a function that takes in a given log-transformed dataset
# and returns an elbow plot of number of principal components versus
# explained variance.

def Elbow_Variance(df):
    k = len(df.axes[1])
    try_n = range(2, k)
    explained_variance = []
    for i in try_n:
        learn = PCA(n_components=i)
        t = learn.fit(df)
        explained_variance.append(sum(t.explained_variance_ratio_))
    
    plt.plot(try_n, explained_variance, "ro")
    plt.show()
    
# This is a function that takes in a dataset (in the uses below, a dataset
# that has been log-transformed and then PCA-transformed and repeatedly performs
# K-Means clustering. It returns an elbow plot of the sum-of-squared errors
# within each cluster versus the number of clusters used.
    
def Cluster_Variance(df):
    # k = len(df[1])
    try_centroids = range(2, 40)
    inertia = []
    for i in try_centroids:
        cluster = KMeans(n_clusters=i, n_init=30)
        c_pca = cluster.fit(df)
        inertia.append(c_pca.inertia_)

    plt.plot(try_centroids, inertia, "ro")
    plt.show()

#### PCA and Clustering
## PCA
Elbow_Variance(logp_new)
# 6 seems to be about the ideal number of principal components. We'll
# transform the dataset into that number of PCs before performing clustering.
pca = PCA(n_components=3)
t = pca.fit_transform(logp_new)

# Plotting 3D PCA.
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30, azim=110)
plt.cla()
ax.scatter(t[:, 0], t[:, 1], t[:, 2], 
           cmap=plt.cm.spectral)
plt.show()

## K-Means clustering with initial parameter
Cluster_Variance(t) # 6 seems to be ideal
pre_cluster = KMeans(n_clusters=4, n_init=30)
pre_c = pre_cluster.fit(t)
pre_labels = pre_c.labels_

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30, azim=110)
plt.cla()
ax.scatter(t[:, 0], t[:, 1], t[:, 2], 
           c=pre_labels.astype(np.float))
plt.show()

## K-Means clustering with revised parameter

## Try mean silhouette score
from sklearn.metrics import silhouette_score
to_try = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
silhouette = []
for i in to_try:
    cluster_try = KMeans(n_clusters=i, n_init=10)
    c_try = cluster_try.fit(t)
    silhouette.append(silhouette_score(t, c_try.labels_))
    
# Now we do a bar chart of the mean silhouette scores by number of clusters
plt.bar(to_try, silhouette, width=1, color="green")

cluster = KMeans(n_clusters=7, n_init=30)
c = cluster.fit(t)
labels = c.labels_

# Let's plot the data with the colors of the clusters.
# Citation for code to color clustering:
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30, azim=110)
plt.cla()
ax.scatter(t[:, 0], t[:, 1], t[:, 2], 
           c=labels.astype(np.float))
plt.show()

#######################################################
#              Jaccardian Similarity                  #
#######################################################

t_df = pd.DataFrame(t)
labels = c.labels_
k = len(c.cluster_centers_)

# Bootstrap of original data
n = t_df.shape[0]
# This is what will be filled with the Jaccard bootstrap values
# j_list = [[], [], [], [], [], [], []]

B = 100
for i in range(0, B):
    #bootstrap = (t_df.sample(n=n, replace=True)).drop_duplicates()
    bootstrap = t_df.sample(n=n, replace=True)
    # http://stackoverflow.com/questions/14661701/how-to-drop-a-list-of-rows-from-pandas-dataframe
    
    lookup = zip(bootstrap.axes[0], range(0, len(bootstrap.axes[0])))
    # This gets overlapping points between original dataset and bootstrap.
    X_star = []
    to_loop = range(0, n)
    for j in to_loop:
        s = t_df.iloc[j]
        def Find_Match(s1, s2=s):
            return s1.equals(s2)
        litmus = bootstrap.apply(func=Find_Match, axis=1)
        if np.mean(litmus) > 0:
            X_star.append(j) 
        elif np.mean(litmus) == 0:
            continue
    
    # This gets En(bootstrap)
    boot_clust = cluster.fit(bootstrap)
    boot_k = len(boot_clust.cluster_centers_)
    boot_labels = boot_clust.labels_

    # This gets all the points in cluster 0 of original data that are also in 
    # X_star.
    # Citation for method to get intersection: 
    # S.Lott's answer here: http://stackoverflow.com/questions/642763/python-intersection-of-two-lists

    # This gets delta and C_i_star
    def Cluster_Overlap(L, k):
        overlap = []
        for j in range(0, k):
            cluster_k = (np.where(L == j))[0]
            overlap.append(set(X_star).intersection(cluster_k))
        return overlap
    
    C_i_star = Cluster_Overlap(labels, k)
    delta = []
    for l in range(0, k):
        grue = (np.where(boot_labels == l))[0]
        cluster_points = []
        temp_points = []
        for m in grue:
            for j in lookup:
                if j[1] == m:
                    cluster_points.append(j[0])
                    delta.append(cluster_points)
    
    if i < 6:
        boot_array = np.array(bootstrap)
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30, azim=110)
        plt.cla()
        ax.scatter(boot_array[:, 0], boot_array[:, 1], boot_array[:, 2], 
           c=boot_labels.astype(np.float))
        plt.show()

    def Jaccard(s1, s2):
        inter = float(len(set(s1).intersection(s2)))
        union = float(len(set(s1).union(s2)))
        return (inter/union)
        
    C_loop = range(0, len(C_i_star))
    delta_loop = range(0, len(delta))
    for j in C_loop:
        similarity = []
        for m in delta_loop:
            similarity.append(Jaccard(C_i_star[j], delta[m]))
        best = np.max(np.array(similarity))
        # j_list[j].append(best)
        
blah = pd.DataFrame(j_list)
tucker = blah.to_csv


def Get_CI(vector):
    s = np.std(vector)
    q = np.percentile(vector, 95)
    x_bar = np.mean(vector)
    CI = [(x_bar - q*s), np.mean(vector), (x_bar + q*s)]
    return CI

  
for i in j_list:
    i.sort()
    print Get_CI(i)
   


################################
#   Debugging Jaccardian Code  #
################################

bootstrap = (t_df.sample(n=n, replace=True)).drop_duplicates()
    # http://stackoverflow.com/questions/14661701/how-to-drop-a-list-of-rows-from-pandas-dataframe

    lookup = zip(bootstrap.axes[0], range(0, len(bootstrap.axes[0])))
    # This gets overlapping points between original dataset and bootstrap.
    X_star = []
    to_loop = range(0, n)
    for j in to_loop: 
        s = t_df.iloc[j]
        def Find_Match(s1, s2=s):
            return s1.equals(s2)
        litmus = bootstrap.apply(func=Find_Match, axis=1)
        if np.mean(litmus) > 0:
            X_star.append(j) 
        elif np.mean(litmus) == 0:
            continue
    
    boot_array = np.array(bootstrap)
    # This gets En(bootstrap)
    boot_clust = cluster.fit(bootstrap)
    boot_k = len(boot_clust.cluster_centers_)
    boot_labels = boot_clust.labels_
    
    # This gets all the points in cluster 0 of original data that are also in 
    # X_star.
    # Citation for method to get intersection: 
    # S.Lott's answer here: http://stackoverflow.com/questions/642763/python-intersection-of-two-lists

    # This gets delta and C_i_star
    def Cluster_Overlap(L, k):
        overlap = []
        for j in range(0, k):
            cluster_k = (np.where(L == j))[0]
            overlap.append(set(X_star).intersection(cluster_k))
        return overlap
    
    C_i_star = Cluster_Overlap(labels, k)
    
    delta = []
    for l in range(0, k):
        grue = (np.where(boot_labels == l))[0]
        cluster_points = []
        for m in grue:
            for j in lookup:
                if j[1] == m:
                    cluster_points.append(j[0])
        delta.append(cluster_points)
                    
    def Jaccard(s1, s2):
        inter = float(len(set(s1).intersection(s2)))
        union = float(len(set(s1).union(s2)))
        return (inter/union)
        
    C_loop = range(0, len(C_i_star))
    delta_loop = range(0, len(delta))
    for j in C_loop:
        similarity = []
        for m in delta_loop:
            similarity.append(Jaccard(C_i_star[j], delta[m]))
        best = np.max(np.array(similarity))
        print best


 

    
    
    
    
    






