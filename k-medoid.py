import numpy
import pandas
import random
import sklearn
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

def cost(center):
    costs = []
    for i in range(len(datapoints)):
        if (i in center):
            continue
        temp = numpy.sum(numpy.absolute(datapoints[center] - datapoints[i]), axis=1)
        costs.append(temp[numpy.argmin(temp)])
    return numpy.sum(costs)

numpy.random.seed(0)
center = numpy.random.choice(len(datapoints), k, replace = False)

finalcenters = center
mincost = cost(center)

for i in range(len(datapoints)):
    if(i in center):
        continue
    newcenter = []
    newcenter.append(i)
    newcenter.append(center[1])
    newcost = cost(newcenter)
    if(newcost<mincost):
        mincost = newcost
        finalcenters = newcenter
    newcenter = []
    newcenter.append(center[0])
    newcenter.append(i)
    newcost = cost(newcenter)
    if(newcost<mincost):
        mincost = newcost
        finalcenters = newcenter

labels = []
for i in range(len(datapoints)):
        temp = numpy.sum(numpy.absolute(datapoints[finalcenters] - datapoints[i]), axis=1)
        labels.append(numpy.argmin(temp))
labels=numpy.array(labels)

print('\n\nNo. of points in each cluster', numpy.bincount(labels))
print('[k-medoid] Cluster centroids -->\n', datapoints[finalcenters])


matplotlib.pyplot.figure(figsize = (18, 15))
plotfig = matplotlib.pyplot.axes(projection='3d')
plotfig.set_title('\nVisual display of %d clusters by k-medoid method: Normalized scale\n' %(k))
# Define lists k no. of colours for displaying data points and centroids
pcolours = ['darkorange', 'cyan']
ccolours = ['red', 'blue']
# Separate data points into k parts based upon their association with respective cluster
for i in range(k): # Selecting columns with high variance for geometrical dispaly
    x = datapoints[labels == i][:, 1] # index = 1 for texture_mean
    y = datapoints[labels == i][:, 11] # index = 11 for texture_se
    z = datapoints[labels == i][:, 13] # index = 13 for area_se
    # Plot data points of all clusters one by one
    plotfig.scatter(x, y, z, s=30, c=pcolours[i], marker='o', label='kM-'+str(i+1)+' Data points')
    # Plot ith centroid
    plotfig.scatter(datapoints[finalcenters][i][1], datapoints[finalcenters][i][11], datapoints[finalcenters][i][13], s=300, c=ccolours[i], marker='<', label='kM-'+str(i+1)+' Centroid')
plotfig.set_xlabel('texture_mean')
plotfig.set_ylabel('texture_se')
plotfig.set_zlabel('area_se')
matplotlib.pyplot.legend()
matplotlib.pyplot.savefig('k-medoid_output.jpg', bbox_inches='tight')
matplotlib.pyplot.show()
