
def extractor(x,y, x_list,log_en):

#    print(x.shape,y.shape)
    
    
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

   
    
    
    """# **특징추출**
    ---

    >>## MFCC
    """

    """전처리 / 특징 추출 """
    """MFCC"""
    #Image('./image/image_09.jpg', width=400, height=100)   # https://blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=sooftware&logNo=221661644808

    # Commented out IPython magic to ensure Python compatibility.
    """library"""
    import librosa
    # %matplotlib inline
    import matplotlib.pyplot as plt
    import librosa.display

    """변수정의"""
    fs=100
    no_of_data=400
    class_list=np.array(['idle', 'suit_1','suit_2','suit_3','suit_4','suit_5','shirt_1','shirt_2','shirt_3',
                'shirt_4','shirt_5','coat_1','coat_2','coat_3','coat_4','coat_5'])

    features_all=pd.DataFrame()

    """함수정의"""
    # list 들어오면, n개씩 쪼개서 return   https://jsikim1.tistory.com/141
    def list_chunk(lst,n):
        return [lst[i:i+n] for i in range(0,len(lst),n)], len(lst)//n      #  // 몫


    for ii in x_list:
    # 각 class별로 n개씩 묶어서(chunk) -->  mfcc -->  merge 함.

    # chunk
        for i in class_list:       # class별로 'idle', 'suit1', ....
            idx_class = y == i
            x_in, count = list_chunk(ii['accx'][idx_class],no_of_data)   # x 400ea
            y_in, count = list_chunk(ii['accy'][idx_class],no_of_data)   # y 400ea
            z_in, count = list_chunk(ii['accz'][idx_class],no_of_data)   # z 400ea
            current_in, count = list_chunk(ii['current'][idx_class],no_of_data)   # current 400ea 
            class_in,count = list_chunk(y[idx_class],no_of_data)   # label
            
            if log_en==1:
                print("class:", i, "    chunk개수:",count)  # class별 chunked 개수 확인

    # mfcc & merge
            for j in range(0,count-1):    # 클래스 내의 j번째 chunk
                # mfcc
                x_in_chunk = np.array(x_in[j])
                y_in_chunk = np.array(y_in[j])
                z_in_chunk = np.array(z_in[j])
                current_in_chunk = np.array(current_in[j])
                #print(x_in_chunk.shape)            # (400,)
                mfccs_x_chunk = librosa.feature.mfcc(x_in_chunk, sr=fs, n_mfcc=13)
                mfccs_y_chunk = librosa.feature.mfcc(y_in_chunk, sr=fs, n_mfcc=13)
                mfccs_z_chunk = librosa.feature.mfcc(z_in_chunk, sr=fs, n_mfcc=13)
                mfccs_current_chunk = librosa.feature.mfcc(current_in_chunk, sr=fs, n_mfcc=13)
                #print(x_in_chunk.shape, mfccs_x_chunk.shape)         # (13,1)

                # class
                class_in_chunk = class_in[0]

                # features merge
                features_chunk = np.concatenate((mfccs_x_chunk,mfccs_y_chunk,mfccs_z_chunk,mfccs_current_chunk),axis=0)

                #features + class
                features_chunk = pd.DataFrame(features_chunk).T
                features_chunk['class']=class_in_chunk
                #features_chunk['class']=class_location

                #print(features_chunk.head())
                features_all = pd.concat([features_all,features_chunk],axis=0)
    

    return features_all

