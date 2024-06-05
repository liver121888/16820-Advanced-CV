# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# 
# Nov, 2023
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import (
    loadData,
    estimateAlbedosNormals,
    displayAlbedosNormals,
    estimateShape,
)
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface

def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    U, S, Vh = np.linalg.svd(I, full_matrices=False)
    S[3:] = 0
    B = Vh[:3, :]
    # L = ((U @ np.diag(S)).T)[:3, :]
    L = U[:3, :]

    print(B.shape, L.shape)
    return B, L


def plotBasRelief(B, mu, nu, lam):
    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    B_prime = np.linalg.inv(G.T) @ B
    albedos, normals = estimateAlbedosNormals(B_prime)
    normals = enforceIntegrability(normals, s)
    surface = estimateShape(normals, s)
    plotSurface(surface)



if __name__ == "__main__":       
    
    I, L0, s = loadData("../data/")

    # Part 2 (b)
    B, L = estimatePseudonormalsUncalibrated(I)

    print(L0)
    print(L)

    # Part 2 (d)
    albedos, normals = estimateAlbedosNormals(B)
    print(normals.shape)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("2d-a.png", albedoIm, cmap="gray")
    plt.imsave("2d-b.png", normalIm, cmap="coolwarm")
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (e)
    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    B_prime = np.linalg.inv(G.T) @ B
    pseudo_normals_e = enforceIntegrability(B_prime, s)
    albedos, normals = estimateAlbedosNormals(pseudo_normals_e)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (f)
    # vary mu
    plotBasRelief(B, -1, 0, -1)
    plotBasRelief(B, 2, 0, -1)
    plotBasRelief(B, 3, 0, -1)

    # vary nu
    plotBasRelief(B, 0, 1, -1)
    plotBasRelief(B, 0, 2, -1)
    plotBasRelief(B, 0, 3, -1)

    # vary lambda
    plotBasRelief(B, 0, 0, -1)
    plotBasRelief(B, 0, 0, 1)
    plotBasRelief(B, 0, 0, 3)

    # plotBasRelief(B, 0, 0, 1e-7)


