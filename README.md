# US-PostalCodes-Clustering

# Machine Learning Clustering Methods: DBSCAN, Hierarchical, K-Mean and Mean Shift

## Clustering US post codes 
The dataset of this project is from  http://www.census.gov/geo/maps-data/data/gazetteer.html, which contains US pos codes information. Longitude and latitude are features used here. Haversine formula is used to calculate the distance between pair of points on the Earth based on their longitude and latitude. To make calculation faster we use 1/10 of data.

![us_poscodes](https://user-images.githubusercontent.com/39537957/50355592-17464500-0504-11e9-837f-802efc5a0e44.png)
To make it  more clear plots don’t show Hawaii and Alaska. 

## DBSCAN 
Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands clusters from them. Since US has 50 states, DBSCAN parameters (eps and min_samples) should be adjust to have around 50 clusters.
![dbscan](https://user-images.githubusercontent.com/39537957/50356806-4c549680-0508-11e9-8763-6ff8488d9a73.png)
Since DBSCAN is good for data which contains clusters of similar density. It is not a good choice for this example. Besides, this algorithm doesn’t assign all the points to clusters label.  As we can see 2681 out of 3314 points are unassigned.

## Hierarchical
Hierarchical clustering creates clusters with top to bottom order. For clustering it is required to determine a distance function to create proximity matrix. We analysis different linkage model to determine which one is suitable. Our choice is Average linkage since the cophenetic correlation coefficient that is generated is the closest to 1.
![hc](https://user-images.githubusercontent.com/39537957/50356895-bb31ef80-0508-11e9-8b9c-9d5fe854ed72.png)
## K-Mean 
K-Mean clustering is pretty fast and needs few computations, since it has linear complexity. In K-Mean clustering the number of clusters should be specify which cannot be consider as a disadvantage for this example. 
![km](https://user-images.githubusercontent.com/39537957/50356908-c84ede80-0508-11e9-8bf6-c8d1681a5a5a.png)
## Mean Shift
Mean Shift is very similar to the K-Means algorithm, except for one very important factor: you do not need to specify the number of groups prior to training. The Mean Shift algorithm finds clusters on its own. On the other hand the main disadvantage of this algorithm is its computational expense for large feature space. It is often slower than K-Mean clustering. Therefore, for this project Mean Shift is not a good choice.   
![ms_clustering](https://user-images.githubusercontent.com/39537957/50357396-bcfcb280-050a-11e9-89a2-0b066c0b2f6b.png)
As we can see it just creates 6 clusters, which is not good for this example.
