from sklearn.datasets import make_blobs
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt
import numpy as np
import random




PROMPT = '''Welcome to the 2-Dimensional K-Means Simulator!\n\n\t[1] Run K-Means\n\t[2] Run Spectral K-Means
\t[3] Run Both\n\t[4] Quit\n\nEnter an option: '''


def generate_data(N, K):
    """This funcyion generates clusterbale data instead of using a uniform
    distribution, so that k-means can do its thing better"""   
    
    # Get those standard deviations
    cluster_std = []
    for k in range(K):
        cluster_std.append(random.randint(0, 100) / 10)
    
    # Make the blobs to get more a clusterable distribution
    center_bb = (0, 100)
    X, y = make_blobs(n_samples=N, 
                      cluster_std=cluster_std, centers=K, 
                      n_features=2, center_box=center_bb)
    return X


def plot_original(X):
    """This function is soley responsible for plotting the original data 
    set BEFORE K-means clusters the data"""
    # Unzip the dataset
    data = list(zip(*X))
    x1, x2 = data[0], data[1]

    # Plots the original dataset
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.title("Original Dataset of Points")
    plt.scatter(x1, x2, marker="o")
    plt.savefig("original_data.png")
    plt.show()


def plot_clusters(K, X, centroids=None):
    # Create the empty clusters
    clusters = []
    for k in range(K):
        clusters.append([])
    
    # Pooling the clusters
    for x in range(len(X)):
       point = X[x][0]
       cluster = X[x][1]
       clusters[cluster].append(point)
        
    # Plots the points of each cluster
    for i, c in enumerate(clusters):
        points = list(zip(*c))
        plt.scatter(points[0], points[1])
        if centroids:
            plt.plot(centroids[i][0], centroids[i][1], marker="x", markersize=7,
                     markeredgecolor="black")
    
    # Makes the plot pretty
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.title("Clustered Dataset")
    if centroids:
        plt.savefig("clustered_data.png")
    else:
        plt.savefig("clustered_data_spectral.png")
    plt.show()
    

def compute_centroid(K, X, sizes, csize):
    # Initialize the centroid array
    centroids = []
    for k in range(K):
        centroids.append(np.array([0] * csize))
    
    # Recompute the centroids
    for x in range(len(X)):
        point = X[x][0]
        cluster = X[x][1]
        centroids[cluster] = np.add(centroids[cluster], point) 
    
    return [c / sizes[i] for i, c in enumerate(centroids)]
    

def KMeans(X, N, K, csize):
    """This function randomly generates n points and will run until K distinct
    partitioned clusters are formed"""
    
    # Initial problem setup
    sizes = [1] * K
    indexes = [i for i in range(N)]
    choices = random.sample(indexes, K)
    centroids = [X[i] for i in choices]
    
    # Initialize the clusters
    clus_num = 0
    X_new = []
    for i in range(len(X)):
        if i in choices:
            X_new.append([X[i], clus_num])
            clus_num += 1
        else:
            X_new.append([X[i], -1])
    X = X_new

    # Cluster the data
    clustering = True
    while clustering:
        clustering = False
        for i in range(len(X)):
            x, c = X[i][0], X[i][1]
            
            # Find the centroid that the point is closest to
            min_dist, min_cluster = sp.euclidean(x, centroids[0]), 0
            for j in range(1, len(centroids)):
                dist = sp.euclidean(x, centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = j

            # Have we re-clustered at all?
            if min_cluster != c:
                clustering = True
                if c != -1:
                    sizes[c] -= 1
                sizes[min_cluster] += 1
            X[i][1] = min_cluster 
                
        # Recompute the centroid
        centroids = compute_centroid(K, X, sizes, csize)
    
    return X, centroids
            

def SpectralKMeans(X, N, K):
    """This function handles the spectral relaxation of k-means"""
    
    # Construct the new matrix of the k largest eigenvectors of X
    XTX = np.matmul(X, np.transpose(X))
    eigenvalues, eigenvectors = np.linalg.eig(XTX)
    
    # Sort to get the largest eigenvectoe/eigenvalue pairs
    # https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt#:~:text=Use%20numpy.,use%20to%20sort%20the%20array.&text=If%20the%20eigenvalues%20are%20complex,broken%20by%20their%20imaginary%20part).
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    eigenvectors = np.transpose(eigenvectors[:K])
    
    # Map the old points to the new points
    X_new = [[0, 0]] * len(X)
    for i in range(len(X)):
        X_new[i] = X[idx[i]]
    X = X_new
    
    # Take the K-dimensional points, cluster them, and bring them back to 2D    
    eigenvectors, centroids = KMeans(eigenvectors, N, K, K)
    for i in range(len(X)):
        eigenvectors[i][0] = X[i]
    return eigenvectors


def main():
    comm = 0
    while comm != 4:
        comm = int(input(PROMPT))
        
        # K-Means
        if comm == 1:
            K = int(input("Enter the number of clusters: "))
            N = int(input("Enter the number of data points: "))
            X = generate_data(N, K)
            plot_original(X)
            X, centroids = KMeans(X, N, K, 2)
            plot_clusters(K, X, centroids)
        
        # Spectral K-Means
        elif comm == 2:
            K = int(input("Enter the number of clusters: "))
            N = int(input("Enter the number of data points: "))
            X = generate_data(N, K)
            plot_original(X)
            X = SpectralKMeans(X, N, K)
            plot_clusters(K, X)
          
        # Test both using the same dataset
        elif comm == 3:
            K = int(input("Enter the number of clusters: "))
            N = int(input("Enter the number of data points: "))
            X = generate_data(N, K)
            plot_original(X)
            X, centroids = KMeans(X, N, K, 2)
            plot_clusters(K, X, centroids)
            X = SpectralKMeans(X, N, K)
            plot_clusters(K, X)
            
        # Error condition
        elif comm != 4:
            print("\nInvalid Command: {}\n".format(comm))


if __name__ == "__main__":
    main()