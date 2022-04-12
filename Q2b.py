import matplotlib.pyplot as plt
from scipy import io
import numpy as np
import random


def plot_image(img, title):
    img = img.reshape(16, 16)
    plt.imshow(img)
    plt.grid(False)
    plt.axis('off')
    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()


def PCA(A, L, p, img):
    """This function hand;es computing PCA on the p prinicipal eigenvectors
    of the covariance matrix of A, using SVD. It will be evaluated using L"""
    
    # DO some SVD to get our eigenvectors
    S, sigma, XT = np.linalg.svd(A, full_matrices=False)
    
    # Get those principal components and plot them
    SK = np.transpose(np.transpose(S)[: p])
    sigmaK = np.diag(sigma[: p])
    XTK = XT[: p]
    AK = np.matmul(np.matmul(SK, sigmaK), XTK)
    plot_image(AK[img], "PCA Image with p={}".format(p))


def main():
    mat = io.loadmat('USPS.mat')
    A = np.array(mat["A"])
    L = np.array(mat["L"])
    img = random.randint(0, 2999)
    plot_image(A[img], "Original Image")
    
    # Go through each of the principal componenets
    for p in [10, 50, 100, 200]:
        PCA(A, L, p, img)


if __name__ == "__main__":
    main()