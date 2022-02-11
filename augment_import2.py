

def augment2(x,y,jitter_sigma,MW_sigma,Scale_sigma,log_en):

    """입출력"""
    import os

    """전처리"""
    import pandas as pd
    import numpy as np
    pd.set_option('display.max_columns', None)

    from scipy.interpolate import CubicSpline      # for Data Augmentation
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    """시각화"""
    import matplotlib.pyplot as plt
    import matplotlib
    plt.style.use('seaborn-whitegrid')
    import seaborn as sns
    #sns.set_style("white")
    import itertools
    # %matplotlib inline

    import warnings
    warnings.filterwarnings(action='ignore')

    
    seed_no = 2022
    np.random.seed(seed = seed_no)

    
    
    
    """jittering"""
    # sigma = 0.005

    def DA_Jitter(X, sigma):
        myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        return X + myNoise

    x_in_jittering = DA_Jitter(x,jitter_sigma)

    
    
    

    """Magnitude Warping"""
    #sigma = 0.05  # default 0.2  (standard deviation)
    knot = 4      # default 4    (should be integer)
    #seed_no = 2022

    ## This example using cubic splice is not the best approach to generate random curves. 
    ## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
    def GenerateRandomCurves(X, sigma, knot=4):
        #np.random.seed(seed = seed_no)
        xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
        x_range = np.arange(X.shape[0])
        cs_x = CubicSpline(xx[:,0], yy[:,0])
        cs_y = CubicSpline(xx[:,1], yy[:,1])
        cs_z = CubicSpline(xx[:,2], yy[:,2])
        cs_a = CubicSpline(xx[:,3], yy[:,3])  # 추가
        cs_b = CubicSpline(xx[:,4], yy[:,4])  # 추가
        cs_c = CubicSpline(xx[:,5], yy[:,5])  # 추가
        cs_i = CubicSpline(xx[:,6], yy[:,6])  # 추가

        return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range),cs_a(x_range),cs_b(x_range),cs_c(x_range),cs_i(x_range)]).transpose()

    
    def DA_MagWarp(X, sigma):
        return X * GenerateRandomCurves(X, sigma)

    ## Random curves around 1.0
    if log_en==1:
        fig = plt.figure(figsize=(16,4))
        for ii in range(8):
            ax = fig.add_subplot(2,4,ii+1)
        #    ax.plot(GenerateRandomCurves(x.iloc[:,0:3], sigma))
            ax.plot(GenerateRandomCurves(x, MW_sigma, knot))
            plt.axis([0,x.shape[0],0.75,1.25])


        fig = plt.figure(figsize=(15,4))
        for ii in range(8):
            ax = fig.add_subplot(2,4,ii+1)
        #    ax.plot(DA_MagWarp(x.iloc[:,0:3], sigma))
            ax.plot(DA_MagWarp(x,MW_sigma))

            # ax.set_xlim([0,20000])
            # ax.set_ylim([-5,5])


    # x 전체 데이터에 Magnitude Warp
    x_in_MagWarp = DA_MagWarp(x,MW_sigma)

    
    
    """>>>### *Scaling*"""

    """Scaling"""
    Scale_sigma_init = 0.025

    def DA_Scaling(X, sigma=Scale_sigma_init):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
        myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
        return X*myNoise

    x_in_Scaling = DA_Scaling(x,Scale_sigma)

    """>>>### *Combination*"""

    """Combination"""
    sigma1 = 0.1
    sigma2 = 0.01

    x_in_Combination = DA_Jitter(DA_Scaling(x,sigma1),sigma2)

    

    
    return x_in_jittering, x_in_MagWarp, x_in_Scaling, x_in_Combination
    
 
