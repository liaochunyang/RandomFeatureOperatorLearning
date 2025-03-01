#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


def RF_Cauchy(gamma, N, x_train, x_test):
    """
    Generate Cauchy Random Features, which approximate Laplace kernel \exp(-\gamma\|x\|_1)
    
    Inputs:
    gamma: scaling parameter of Cauchy distribution.
    N: number of random features
    x_train: training samples of shape m x d
    x_test: test samples of shape m' x d
    
    Outputs:
    A_train: Training Random Feature Map A_train
    A_test: Test random feature map A_test
    """
    # number of samples and dimension of features
    m,d = x_train.shape
    # random features
    Omega = gamma*np.random.standard_cauchy(size = (d,N))
    
    # Random feature matrix A
    random_offset = np.random.uniform(0, 2 * np.pi, size=(1,N))
    A_train = np.cos(x_train@Omega + random_offset)
    A_test = np.cos(x_test@Omega + random_offset)
    
    return A_train * (2.0 / N) ** 0.5, A_test * (2.0 / N) ** 0.5


def RF_Gaussian(gamma, N, x_train, x_test):
    """
    Generate Gaussian Random Features, which approximate Gaussian kernel \exp(-\gamma\|x\|_2^2)
    
    Inputs:
    gamma: (2*gamma)**0.5 is the variance of the Gaussian distribution
    N: number of random features
    x_train: training samples of shape m x d
    x_test: test samples of shape m' x d
    
    Outputs:
    A_train: Training Random Feature Map A_train
    A_test: Test random feature map A_test
    """
    # number of samples and dimension of features
    m,d = x_train.shape
    # random features
    Omega = (2.0 * gamma) ** 0.5*np.random.normal(size = (d,N))
    
    # Random feature matrix A
    random_offset = np.random.uniform(0, 2 * np.pi, size=(1,N))
    A_train = np.cos(x_train@Omega + random_offset)
    A_test = np.cos(x_test@Omega + random_offset)
    
    return A_train * (2.0 / N) ** 0.5, A_test * (2.0 / N) ** 0.5


def student(nu, sigma, size):
    """
    Generate student t distributions.
    Args:
    nu: degree of freedom of chi-square distribution.
    sigma: standard deviation of normal distribution.
    size: size of output samples.
    Returns:
    samples of student t distribution of given size.
    """
    
    gaussian = np.random.normal(loc=0, scale=sigma, size=size)
    chisquare = np.random.chisquare(nu, size=(1,size[1]))
    
    return np.sqrt(nu/chisquare)*gaussian


def RF_student(nu, gamma, N, x_train, x_test):
    """
    Generate Student Random Features, which approximate Matern kernel 
    
    Inputs:
    nu: 
    gamma: scaling parameter of Cauchy distribution.
    N: number of random features
    x_train: training samples of shape m x d
    x_test: test samples of shape m' x d
    
    Outputs:
    A_train: Training Random Feature Map A_train
    A_test: Test random feature map A_test
    """
    # number of samples and dimension of features
    m,d = x_train.shape
    # random features generated from Cauchy distribution with scaling parameter gamma
    Omega = student(nu,gamma,size=(d,N))
    # Random feature matrix A
    random_offset = np.random.uniform(0, 2 * np.pi, size=(1,N))
    A_train = np.cos(x_train@Omega + random_offset)
    
    A_test = np.cos(x_test@Omega + random_offset)
    
    return A_train * (2.0 / N) ** 0.5, A_test * (2.0 / N) ** 0.5

