import pandas as pd
import numpy as np
from sklearn import cluster, metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
matplotlib.style.use('ggplot') 
from sklearn import datasets

from math import radians, cos, sin, asin, sqrt

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

# US postal code info are from http://www.census.gov/geo/maps-data/data/gazetteer.html
X = pd.read_csv('postcodes_US_R.csv')

im = plt.imread("blank-united-states-map11.png")
plt.imshow(im, zorder=1, extent=[-60,0, 21, 50])

# US postcodes plot. To make it more clear it doesn't show Alaska and Hawaii.If you want to see the whole map on this plot just remove xlim and ylim 
plt.scatter(X["Longitude"], X["Latitude"], zorder=0)
plt.ylim(20,55)
plt.xlim(-130,0)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# calculates the distance between any points on the Earth's surface specified by their longitude and latitude
def haversine(lonlat1, lonlat2):
    """
    Calculates the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2.)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.)**2
    c = 2 * asin(sqrt(a)) 
    r = 3959 # approximate radius of earth in miles.
    return c * r

# distance matrix between pairs of points
my_metric = pdist(X.loc[:,['Latitude','Longitude']], metric=(lambda u,v: haversine(u,v)))
distance_matrix = squareform(my_metric)
print("Distance matrix between pairs of points.")
print ("Distance matrix shape:",distance_matrix.shape)
print ("Distance matrix min:",distance_matrix.min())
print ("Distance matrix mean:",distance_matrix.mean())
print ("Distance matrix max:",distance_matrix.max())

"""""""""""""""""
DBSCAN Clustering

input: latitude and longitude matrix
feature array: the distance matrix
eps and min_samples: try a range of number to find the best eps and min_samples that give us suitable number of clusters
"""""""""""""""""
# to gain 50 clusters, number of US states, try a ranhe of numbers to find the aproprate eps and min_samples
listy = []
for i in [5,10,15,20]:
    for j in range(1,10,1):
        db = DBSCAN(eps=i, min_samples=j, metric='precomputed')
        y_db = db.fit_predict(distance_matrix)
        if len(set(y_db))>30 and len(set(y_db))<100:
            print (i,j,len(set(y_db)))
            listy.append([i,j,len(set(y_db))])
             
db = DBSCAN(eps=15, min_samples=6, metric='precomputed')  
y_db = db.fit_predict(distance_matrix)
print ("Epsilon = 15 and min_samples = 6 results in",len(set(y_db)),"clusters.")

plt.imshow(im, zorder=1, extent=[-60,0, 21, 50])

plt.scatter(X["Longitude"], X["Latitude"], c=y_db, cmap='rainbow')
plt.ylim(20,55)
plt.xlim(-130,0)
plt.title("Epsilon = 15 and min_samples = 6")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

print (pd.DataFrame(y_db)[0].value_counts()[-1], "out of",len(y_db),"points are unassigned.")

"""""""""""""""""""""""
Hierarchical Clustering

input: latitude and longitude matrix
linkage function: the distance matrix
cophenetic coefficient: Harversine formula metric 
"""""""""""""""""""""""
my_metric = pdist(X.loc[:,['Latitude','Longitude']], metric=(lambda u,v: haversine(u,v)))

# to find out what is the best linkage methods different methods are tried here and avarage method generates the closest Cophenetic to 1
Z = linkage(distance_matrix,'single')
c, coph_dists = cophenet(Z,my_metric)
print("coph_dists with single",c)
Z = linkage(distance_matrix,'complete')
c, coph_dists = cophenet(Z,my_metric)
print("coph_dists with complete",c)
Z = linkage(distance_matrix,'ward')
c, coph_dists = cophenet(Z,my_metric)
print("coph_dists with ward",c)
Z = linkage(distance_matrix,'average')
c, coph_dists = cophenet(Z,my_metric)
print("coph_dists with average",c)

# to get suitable number of clusters a range of numbers is tried here to find the best threshold
for i in [8000,8500,8700,8920,9000]:
    max_d = i
    clusters = fcluster(Z, max_d, criterion='distance')
    print (i, len(set(clusters)))

max_d = 8920
clusters = fcluster(Z, max_d, criterion='distance')
print ("Hierarchical clustering with max_d =",max_d,"results in",len(set(clusters)),"clusters.")

plt.imshow(im, zorder=1, extent=[-60,0, 21, 50])

plt.scatter(X["Longitude"], X["Latitude"], c=clusters, cmap='prism')
plt.ylim(20,55)
plt.xlim(-130,0)
plt.title("Hierarchical clustering with max_d ="+str(max_d))
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

"""""""""""""""""
KMeans Clustering

number of cluster: number of US states (50)
feature array: the distance matrix
"""""""""""""""""
kmeans = KMeans(n_clusters=50)
kmeans.fit(distance_matrix)

plt.imshow(im, zorder=1, extent=[-60,0, 21, 50])

plt.scatter(X["Longitude"], X["Latitude"], c=kmeans.labels_, cmap='rainbow', zorder=0)
plt.ylim(20,55)
plt.xlim(-130,-60)
plt.title("KMeans clustering k=50")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

"""""""""""""""""
Mean_shift Clustering

"""""""""""""""""
#bandwidth = estimate_bandwidth([X["Longitude"], X["Latitude"]])
meanshift = MeanShift()
meanshift.fit(distance_matrix)

print("number of estimated clusters in Mean Shift Clustering",len(np.unique(meanshift.labels_)))

plt.scatter(X["Longitude"], X["Latitude"], c= meanshift.labels_, cmap='rainbow')
#plt.ylim(20,55)
#plt.xlim(-130,-60)
plt.title("MeanShift Clustering")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

