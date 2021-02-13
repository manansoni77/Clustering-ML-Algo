import numpy
import random
import sklearn
import pandas
import matplotlib
import matplotlib.pyplot
import sklearn.preprocessing

# input and process data
originaldata = pandas.read_csv('cancer.csv')
del originaldata["id"]
del originaldata["diagnosis"]
del originaldata["Unnamed: 32"]
scaledata = sklearn.preprocessing.StandardScaler().fit_transform(originaldata.values)
normaldata = sklearn.preprocessing.normalize(scaledata, norm = 'l2')

datapoints = normaldata
k = 2

numpy.random.seed(0)
previousseeds = datapoints[numpy.random.choice(len(datapoints), k)]
currentseeds = []
labels = []

while True:
    temp = []
    for i in range(len(datapoints)):
        temp.append(numpy.argmin(numpy.sqrt(numpy.sum((previousseeds - datapoints[i])**2, axis=1))))
    labels = numpy.array(temp)
    temp = [] 
    for j in range(k):
        temp.append(numpy.mean(datapoints[labels==j], axis = 0))
    currentseeds = numpy.array(temp)
    if numpy.all(previousseeds == currentseeds):
        break
    previousseeds = currentseeds

print('\n\nNo. of points in each cluster', numpy.bincount(labels))
print('[k-means] Cluster centroids -->\n', currentseeds)

matplotlib.pyplot.figure(figsize = (18, 15))
plotfig = matplotlib.pyplot.axes(projection='3d')
plotfig.set_title('\nVisual display of %d clusters by k-means method: Normalized scale\n' %(k))

# Define lists of colors for plotting points and centers
pcolours = ['darkorange', 'cyan']
ccolours = ['red', 'blue']

# Separate data points into k parts based upon their association with respective cluster
for i in range(k): # Selecting columns with high variance for geometrical dispaly
    x = datapoints[labels == i][:, 1] # index = 1 for texture_mean
    y = datapoints[labels == i][:, 11] # index = 11 for texture_se
    z = datapoints[labels == i][:, 13] # index = 13 for area_se
    # Plot data points of all clusters one by one
    plotfig.scatter(x, y, z, s=30, c=pcolours[i], marker='o', label='kM-'+str(i+1)+' Data points')
    # Plot centroid
    plotfig.scatter(currentseeds[i][1], currentseeds[i][11], currentseeds[i][13], s=300, c=ccolours[i], marker='<', label='kM-'+str(i+1)+' Centroid')
plotfig.set_xlabel('texture_mean')
plotfig.set_ylabel('texture_se')
plotfig.set_zlabel('area_se')
matplotlib.pyplot.legend()
matplotlib.pyplot.savefig('k-means_output.jpg', bbox_inches='tight')
matplotlib.pyplot.show()
