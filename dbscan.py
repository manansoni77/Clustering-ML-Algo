import numpy
import matplotlib.pyplot
import pandas
import sklearn.preprocessing
data = pandas.read_csv('cancer.csv')
del data["id"]
del data["diagnosis"]
del data["Unnamed: 32"]
data = sklearn.preprocessing.StandardScaler().fit_transform(data)
data = sklearn.preprocessing.normalize(data)
data = data[:, [1,11,13]]
eps = 0.2
minpts = 3
def update_labels(X,pt,eps,labels,cluster_val,minpts):
    neighbors = []
    label_index = []
    for i in range(X.shape[0]):
        if numpy.linalg.norm(X[pt]-X[i])<eps:
            neighbors.append(X[i])
            label_index.append(i)
    if len(neighbors)==0:
        labels[p] = -1
    elif len(neighbors) <minpts:
        for i in range(len(labels)):
            if i in label_index:
                labels[i]=cluster_val
    else:
        for i in range(len(labels)):
            if i in label_index:
                labels[i]=cluster_val
    return labels
labels = [0]*data.shape[0]
C = 1
for p in range(data.shape[0]):
    if labels[p]==0:
        labels = update_labels(data,p,eps,labels,C,minpts)
        C= C+1
matplotlib.pyplot.figure(figsize = (18, 15))
ax = matplotlib.pyplot.axes(projection='3d')
ax.set_title('\nVisual display of clusters by DBScan method: Normalized scale\n')
x = data[:, 0] # index = 1 for texture_mean
y = data[:, 1] # index = 11 for texture_se
z = data[:, 2] # index = 13 for area_se
# Plot data points of all clusters one by one
ax.scatter(x, y, z, s=30, c=labels, cmap = 'inferno')
ax.set_xlabel('texture_mean')
ax.set_ylabel('texture_se')
ax.set_zlabel('area_se')
matplotlib.pyplot.savefig('dbscan_output(eps=0.2,minpts=3).jpg', bbox_inches='tight')
matplotlib.pyplot.show()
