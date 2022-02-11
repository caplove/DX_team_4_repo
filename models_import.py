

def my_models(features_all,log_en):

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


    seed_no = 0


    """# **모형 설계/훈련/평가**
    ---

    >>>### x,y 정의 (features 기반)
    """

    """x,y 정의"""
    x=features_all.drop('class',axis=1)
    y=features_all['class']
    class_list=np.array(['idle', 'suit_1','suit_2','suit_3','suit_4','suit_5','shirt_1','shirt_2','shirt_3',
                'shirt_4','shirt_5','coat_1','coat_2','coat_3','coat_4','coat_5'])
    """class를 숫자로 변환 for pytorch"""
    y_number=[]
    for i in y:
        y_tmp=(np.where(i == class_list))[0][0]    # [0][0] 추가해서 데이터만 추출
        #print (i,y_tmp)
        y_number.append(y_tmp)
        #print(i,y_tmp[0][0])

    y=np.array(y_number).reshape(-1,)
    y=pd.Series(y)

    if log_en==1:
        # 데이터 shape check  (AAA,00) (000,)
        print(x.shape,y.shape)

        display(x.head())
        display(y.head())

    """>>## 데이터셑/Scaler"""

    """데이터셑 나누기"""
    # train, test
    train_x, test_x, train_y, test_y = train_test_split(x, y, stratify=y, test_size=0.2,random_state=seed_no)
    # train, validation
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, stratify=train_y, test_size=0.2,random_state=seed_no)
    
    if log_en==1:
        print(f"학습 데이터셋 크기 : {train_x.shape}, 검증 데이터셋 크기 : {valid_x.shape}, 테스트 데이터셋 크기 : {test_x.shape}")

    """scaler"""
    scaler=StandardScaler()
    scaler.fit(train_x)
    train_x = pd.DataFrame(scaler.transform(train_x))
    valid_x = pd.DataFrame(scaler.transform(valid_x))
    test_x = pd.DataFrame(scaler.transform(test_x))

    if log_en==1:

        pd.DataFrame(train_x, columns = x.columns).describe()
        #평균과 STD 확인

        plt.figure(figsize=(10, 6))

        plt.subplot(311)
        plt.hist(train_y, bins=np.arange(0, 16, 0.5))
        plt.ylim(0, 150)
        plt.xticks(np.arange(0 , 11, 1))
        plt.title('Training Data')

        plt.subplot(312)
        plt.hist(valid_y, bins=np.arange(0, 16, 0.5))
        plt.ylim(0, 150)
        plt.xticks(np.arange(0 , 11, 1))
        plt.title('Validation Data')

        plt.subplot(313)
        plt.hist(test_y, bins=np.arange(0, 16, 0.5))
        plt.ylim(0, 150)
        plt.xticks(np.arange(0 , 11, 1))
        plt.title('Testing Data')
        plt.show()

    """>>## DNN

    >>>### 분류기 모형
    """

    """분류기 모형 설정"""

    """DNN"""
    '''Neural Network을 위한 딥러닝 모듈'''
    import torch             # 딥러닝 모듈이고, 로컬에서는  설치필요
    import copy
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ''' 결과 평가용 모듈 '''
    from sklearn.metrics import accuracy_score, confusion_matrix

    ''' 기타 optional'''
    import warnings, itertools
    warnings.filterwarnings(action='ignore')


    #https://colab.research.google.com/drive/1kt8Zmy-qiE4mhcgBrFUAzdBmVh9UVyGK#scrollTo=1ed897f1
    train_x_torch = torch.FloatTensor(train_x.values) # torch.FloatTensor(numpy)
    train_y_torch = torch.LongTensor(train_y.values) # torch.LongTensor(numpy)
    trainDataset = torch.utils.data.TensorDataset(train_x_torch, train_y_torch)
    trainLoader = torch.utils.data.DataLoader(dataset = trainDataset,
                                             batch_size = 100,
                                             shuffle = True)

    # 검증에는 shuffle 하지 않음.  w 찾는게 아님.
    valid_x_torch = torch.FloatTensor(valid_x.values) # torch.FloatTensor(numpy)
    valid_y_torch = torch.LongTensor(valid_y.values) # torch.LongTensor(numpy)
    validDataset = torch.utils.data.TensorDataset(valid_x_torch, valid_y_torch)
    validLoader = torch.utils.data.DataLoader(dataset = validDataset,
                                            batch_size = 100,
                                            shuffle = False)

    test_x_torch = torch.FloatTensor(test_x.values) # torch.FloatTensor(numpy)
    test_y_torch = torch.LongTensor(test_y.values) # torch.LongTensor(numpy)
    testDataset = torch.utils.data.TensorDataset(test_x_torch, test_y_torch)
    testLoader = torch.utils.data.DataLoader(dataset = testDataset,
                                            batch_size = 100,
                                            shuffle = False)

    # 모형
    class DNNClassifier(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):    # 모델 구조 정의
            # input_dim = 784, output_dim = 10 (클래스 개수)
            super().__init__()

            ''' 모델 구조 만들기'''
            # CNN, RNN, LSTM등은 linear가 아니다.
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim1) # input_dim(784) -> hidden_dim1(500)
            self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2) # hidden_dim1(500) -> hidden_dim2(300)
            self.linear3 = torch.nn.Linear(hidden_dim2, hidden_dim3) # hidden_dim2(300) -> hidden_dim3(100)
            self.linear4 = torch.nn.Linear(hidden_dim3, output_dim) # hidden_dim3(100) -> output_dim(10)

            self.relu = torch.nn.ReLU() # Relu activation function
            self.dropout = torch.nn.Dropout(p=0.5)     # p = 0.5는 확률개념으로 50% 는 w 를 업데이트 하지 않음. overfit 방지

        def forward(self, x):                                                                # 순서, Sequence 정의

            ''' 짜여진 모델에 설명 변수 데이터 x를 입력할 때 진행할 순서 설정'''

            x = self.linear1(x) 
            x = self.relu(x) 
            x = self.linear2(x)
            x = self.relu(x)
            x = self.dropout(x)     # 반절만 활성화
            x = self.linear3(x)
            x = self.relu(x)
            output = self.linear4(x) 

            return output

    #모델 및 비용함수, Solver 설정
    if log_en==1:
        print(train_x.shape[1],train_y.nunique())

    # 빈 모델 생성
    clf_DNN = DNNClassifier(train_x.shape[1], 500, 300, 100, train_y.nunique())  # features개수, Hidden Layer1, Hidden Layer2, Hidden Layer3, class 개수

    # 비용함수 정의
    criterion = torch.nn.CrossEntropyLoss()

    # Solver 설정
    # 경사하강법의 종류 정의 (adam)
    solver = torch.optim.Adam(clf_DNN.parameters(), lr = 0.001)

    """>>>### 훈련 / 검증"""

    num_epochs = 100

    if log_en==1:
        print("Start Training !")
        print('-'*50)

    # 빈 공간 할당
    train_loss_total = []
    valid_loss_total = []
    best_loss = np.inf


    # 
    for epoch in range(num_epochs):
        #초기값  
        train_loss = 0
        valid_loss = 0

        ''' Training '''
        clf_DNN.train()
        for x_data, y_data in trainLoader:          # trainLoader로 구성해놨던 데이터 가져옴.

            # 정확한 학습을 위하여 모든 기울기 값을 0으로 설정
            solver.zero_grad()

            y_pred = clf_DNN(x_data)

            # 비용함수를 활용하여 오차 계산
            loss = criterion(y_pred, y_data)

            # 계산된 오차를 기반으로, 오차를 줄일 수 있는 방향으로 w값 업데이트  (즉, backpropagation !)
            loss.backward()
            solver.step() # forward evaluation, backward propagation, update를 모두 포함하는 step

            train_loss += loss.item()

        ''' Validation '''
        clf_DNN.eval()
        for eval_x_data, eval_y_data in validLoader:
            eval_y_pred = clf_DNN(eval_x_data)
            valid_loss += criterion(eval_y_pred, eval_y_data).item() # 딕셔너리에 있는 키와 값들의 쌍을 얻어 저장



        """ 결과출력"""
        if log_en==1:
            print('[%d epoch] Train loss : %.3f, Valid loss : %.3f' % (epoch+1, train_loss/len(trainLoader), valid_loss/len(validLoader)))

        if valid_loss/len(validLoader) < best_loss:
            # 로스값 업데이트
            best_loss = valid_loss/len(validLoader)        # validation loss 값이 점점줄다가 다시커지게 되므로 가장적었을때의 epoch와 파라미터 w를 기억해둔다.
            # 최적의 epoch 수와 모델 저장하기
            best_epoch = epoch
            best_model = clf_DNN.state_dict()

        train_loss_total.append(train_loss/len(trainLoader))
        valid_loss_total.append(valid_loss/len(validLoader))

    if log_en==1:  
        print('-'*50)
        print("Finished Training ! Best Epoch is epoch %d." % (best_epoch+1))

        # learning Curve
        plt.figure(figsize=(15,7))

        # 학습 및 검증 로스 변동 관찰하기
        plt.plot(train_loss_total,label='Train Loss')
        plt.plot(valid_loss_total, label='Validation Loss')
        # 최적의 모델이 저장된 곳 표시
        plt.axvline(x = best_epoch, color='red', label='Best Epoch')    # axv 는 수직선 그리기
        plt.legend(fontsize=15)
        plt.title("Learning Curve of trained DNN Classifier", fontsize=18)
        plt.show()

    # 빨간선왼쪽, underfit, 오르쪽은 overfit
    # 주관적으로 약간 왼쪽 또는 오른쪽으로 볼수 있음.

    """1차 평가 /w test_set"""
    # DNN 성능평가
    # 최적의 모델 불러오기
    best_clf_DNN = DNNClassifier(train_x.shape[1], 500, 300, 100, 16)
    best_clf_DNN.load_state_dict(best_model)

    # model을 evaluation 모드로 변경
    best_clf_DNN.eval()

    # clf_mlp(data) == data -> logit -> probability=softmax(logit)
    y_train_prob = best_clf_DNN(train_x_torch).softmax(dim=1)

    # 가장 큰 확률값에 해당하는 범주를 예측 범주로 저장
    y_train_pred = y_train_prob.max(1)[1].numpy()

    y_train_prob[0].detach().numpy().tolist()

    y_train_pred[0]

    train_y.head(1)

    train_accuracy = accuracy_score(y_pred=y_train_pred,y_true=train_y)

    if log_en==1:
        print(f"훈련 데이터셋 정확도: {train_accuracy:.3f}")

    cm_train = confusion_matrix(y_true=train_y, y_pred=y_train_pred)

    if log_en==1:
        plt.figure(figsize=(8, 8))
        sns.heatmap(data=cm_train, annot=True, fmt='d', annot_kws={'size': 18}, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    """학습된 DNN Classifier 결과 확인 및 성능 평가 : Validation Data"""
    y_valid_prob = best_clf_DNN(valid_x_torch).softmax(dim=1)
    y_valid_pred = y_valid_prob.max(1)[1].numpy()
    valid_accuracy = accuracy_score(y_pred=y_valid_pred,y_true=valid_y)

    if log_en==1:
        print(f"검증용 데이터셋 정확도: {valid_accuracy:.3f}")

    cm_valid = confusion_matrix(y_true=valid_y, y_pred=y_valid_pred)
    
    if log_en==1:
        plt.figure(figsize=(8, 8))
        sns.heatmap(data=cm_valid, annot=True, fmt='d', annot_kws={'size': 18}, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    """성능평가 : Testing Data"""
    y_test_prob = best_clf_DNN(test_x_torch).softmax(dim=1)
    y_test_pred = y_test_prob.max(1)[1].numpy()
    Acc_DNN = accuracy_score(y_pred=y_test_pred,y_true=test_y)

    if log_en==1:
        print(f"테스트용 데이터셋 정확도: {Acc_DNN:.3f}")

    cm_test = confusion_matrix(y_true=test_y, y_pred=y_test_pred)
    
    if log_en==1:
        plt.figure(figsize=(8, 8))
        sns.heatmap(data=cm_test, annot=True, fmt='d', annot_kws={'size': 18}, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    """>>## MLP
    - <b>activation</b> - activation function 타입 (identity, logistic, tanh, <font color='red'>relu</font>) <br>
    - <b>batch_size</b> - stochastic optimizer가 사용할 minibatch 크기 <br>
    - <b>max_iter  </b> - stochastic optimizer의 최대 iteration 횟수 ( = Epochs )<br>
    - <b>alpha     </b> - Learning Rate (과적합 방지용) <br>
    - <b>solver    </b> - 경사하강법의 종류 (<font color='red'>adam</font>, sgd, lbfgs) <br>

    -- lbfgs: L-BFGS 준-뉴턴 방식 의 최적화 알고리즘으로, 제한된 컴퓨터 메모리를 이용하여 기존 BFGS 알고리즘을 속도면에서 개선한 알고리즘
    """

    ''' Neural Network Classifier(분류기) 모듈 '''
    from sklearn.neural_network import MLPClassifier    # MultiLayerPerceptrion

    ''' 결과 평가용 모듈 '''
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score   # 분류
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error       # 예측

    """>>>### 분류기 모형"""

    """분류기 모형 설정"""
    clf_mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size=10,        # 몇 개의 data를 보고 학습할 것인지. batch_size
                            hidden_layer_sizes=(32), max_iter=500,                 
                            solver='adam', verbose = log_en, random_state = 2022)

    """>>>### 훈련 / 검증"""

    """학습"""
    clf_mlp.fit(train_x, train_y)

    """1차 평가 /w training_set"""
    if log_en==1:

        plt.figure(figsize=(20,10))

        train_loss_values = clf_mlp.loss_curve_
        plt.plot(train_loss_values,label='Train Loss')

        plt.legend(fontsize=20)
        plt.title("Learning Curve of trained MLP Classifier", fontsize=18)
        plt.show()

    train_y_pred = clf_mlp.predict(train_x)

    cm_train = confusion_matrix (y_true=train_y, y_pred=train_y_pred)

    if log_en==1:

        plt.figure(figsize=(10, 10))
        sns.heatmap(data=cm_train, annot=True, fmt='d', annot_kws={'size': 12}, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    """2차 평가 /w testing_set"""
    test_y_pred = clf_mlp.predict(test_x)

    cm_test = confusion_matrix(y_true=test_y, y_pred=test_y_pred)

    if log_en==1:
    
        plt.figure(figsize=(10, 10))
        sns.heatmap(data=cm_test, annot=True, fmt='d', annot_kws={'size': 18}, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    test_y_pred_proba = clf_mlp.predict_proba(test_x)
    test_y_pred_proba = pd.DataFrame(test_y_pred_proba)
    test_y_pred_proba.index = test_y.index.copy()

    test_results = pd.concat([test_y_pred_proba, test_y], axis=1)
    test_results.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    Acc_MLP = accuracy_score(test_y,test_y_pred)
    #test_results.head()

    if log_en==1:

        print("정확도 : {:.3f}".format(accuracy_score(test_y,test_y_pred)))
        print("오차 행렬 \n",confusion_matrix(test_y,test_y_pred))

        print(classification_report(test_y,test_y_pred))

    """>>## Decision Tree
    - 최적의 max_depth를 선택해야 함
    - Validation을 사용하여 accuracy, F1-Score를 고려하여 선정
    """

    """라이브러리 모형 설정"""

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix     # f1_score 는 극단적 데이터 불균형이 있을경우에 살펴봐야 함.
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree

    max_depths = list(range(1, 20)) + [None]

    if log_en==1:
        print(max_depths)

    # 평가 지표 저장
    acc_valid = []
    f1_valid = []

    """>>>### 분류기 모형 / 훈련"""

    """1차 평가 /w training set"""
    for max_depth in max_depths:

        # 모델 학습
        model = DecisionTreeClassifier(max_depth=max_depth,random_state=seed_no)
        model.fit(train_x, train_y)

        # validation 예측
        y_valid_pred = model.predict(valid_x)

        # 모델 평가 결과 저장
        acc = accuracy_score(valid_y, y_valid_pred)
        #f1 = f1_score(valid_y, y_valid_pred,average='macro')

        acc_valid.append(acc)   # 덮어씌워지므로 apped
        #f1_valid.append(f1)     # append

   
    if log_en==1:
    
        # Decision Tree depth list
        xticks = list(map(str, max_depths))     # 스트링을 그래프의 문자열로 사용하려면.
        print(xticks)

        # Decision Tree depth에 따른 accuracy
        fig, ax = plt.subplots(figsize=(15, 6))
        #fig.subplots_adjust(right=0.75)

        ax.plot(range(len(max_depths)), acc_valid, color='red', marker='o')
        ax.set_ylabel('accuracy', color='red', fontsize=12)

        # ax2 = ax.twinx()
        # ax2.plot(range(len(max_depths)), f1_valid, color='blue', marker='s')
        # ax2.set_ylabel('f1', color='blue', fontsize=12)

        plt.xticks(range(len(max_depths)), xticks)
        plt.show()

    model = DecisionTreeClassifier(max_depth=8,random_state=seed_no)
    model.fit(train_x, train_y)

    """>>>### 검증"""

    """1차 평가 /w test_set"""
    y_test_pred = model.predict(test_x)

    # Confusion Matrix
    cm = confusion_matrix(test_y, y_test_pred)
    cm = pd.DataFrame(cm)

    # Accuracy, F1-Score
    Acc_TREE = accuracy_score(test_y, y_test_pred)
    f1 = f1_score(test_y, y_test_pred, average='macro')

    if log_en==1:

        print('- Accuracy (Test) : {:.3}'.format(Acc_TREE))
        print('- F1 score (Test) : {:.3}'.format(f1))

        # 시각화
        plt.figure(figsize=(10, 10))
        sns.heatmap(data=cm, annot=True, annot_kws={'size': 15}, fmt='d', cmap='Blues')
        plt.title('Acc = {:.3f} & F1 = {:.3f}'.format(Acc_TREE, f1))
        plt.show()

        plt.figure(figsize=(25, 15))
        plot_tree(decision_tree=model,max_depth=3,feature_names=None,label='all', filled=True,fontsize=12)
        plt.show()
        # Samples = 3  샘플의 개수
        # Values = [2,1]     색상

    # 변수 중요도
    importances = model.feature_importances_

    # 내림차순으로 정렬하기 위한 index
    index = np.argsort(importances)[::-1]

    if log_en==1:

        plt.figure(figsize=(25, 6))
        plt.title('Feature Importances')
        plt.bar(range(x.shape[1]),
                importances[index],
                align='center')
        plt.xticks(range(x.shape[1]), x.columns[index], rotation=90)
        plt.xlim([-1, x.shape[1]])
        plt.show()

    """>>## KNN
    grid search의 대상이 되는 파라미터
    - n_neighbors: 근접이웃 개수
    - weights: weight for voting
    - metric: 거리 계산 방법
    """

    """ 모델 생성, 학습, 평가 """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix    # 평가법은 acc,f1,confusion matrix
    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.neighbors import VALID_METRICS
    from sklearn.metrics import SCORERS

    # parameter grid 지정                 5-fold / 10-fold 주로 사용
    # dictioanry 안의 list 형태로 넣어서 구성

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['cosine', 'euclidean', 'manhattan']
    }

    """>>>### 분류기 모형 / 훈련"""

    """ Grid Search CV 모델 구성: cv = 10 """

    SCORERS.keys()
    model = KNeighborsClassifier()
    model_cv = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=10,
                            n_jobs=-1)  # 모든것을 고려하겠다...


    model_cv.fit(train_x, train_y)

    results = model_cv.cv_results_
    params = results['params']
    mean_score = results['mean_test_score']
    std_score =results['std_test_score']

    params = ['_'.join(str(x) for x in param.values()) for param in params]
    params[:5]

    if log_en==1:

        plt.figure(figsize=(15, 6))
        plt.fill_between(range(len(params)), mean_score - std_score, mean_score + std_score, alpha=0.1, color='blue')
        plt.plot(range(len(params)), mean_score, color='blue')
        plt.ylim([0.7, 1.0])
        plt.xticks(range(len(params)), params, rotation=90)
        plt.show()

        # Best parameter 출력
        print('Best parameters (Grid Search) \n >> ', model_cv.best_params_)

    # Best 모델 찾기
    model = model_cv.best_estimator_
    model                                         # weight default와 같아서 안보임.

    """>>>### 검증"""

    # 에측 결과 산출
    y_test_pred = model.predict(test_x)

    # Confusion Matrix
    cm = confusion_matrix(test_y, y_test_pred)
    cm = pd.DataFrame(cm)

    # Accuracy, F1-Score
    Acc_KNN = accuracy_score(test_y, y_test_pred)
    f1 = f1_score(test_y, y_test_pred,average='macro')

    if log_en==1:
    
        print('- Accuracy (Test) : {:.3}'.format(Acc_KNN))
        print('- F1 score (Test) : {:.3}'.format(f1))

        # 시각화
        plt.figure(figsize=(10, 10))
        sns.heatmap(data=cm, annot=True, annot_kws={'size': 15}, fmt='d', cmap='Blues')
        plt.title('Acc = {:.3f} & F1 = {:.3f}'.format(Acc_KNN, f1))
        plt.show()

    """>>## **Ensemble(AdaBoost)**

    >>>### 분류기 모형 / 훈련
    """

    from sklearn.ensemble import AdaBoostClassifier

    params = {"n_estimators" : [100, 200, 400], "learning_rate" : [0.01, 0.5, 1.0]}

    # model define
    model_Ada = AdaBoostClassifier(random_state=seed_no)

    # model train with gridsearchCV    # CV = CrossValidation    5개의 데이터셑으로 분할하고, 4번 훈련, 1번 test 수행 
    grid_model_Ada = GridSearchCV(model_Ada, param_grid = params, cv = 5, refit = True, return_train_score= True)
    grid_model_Ada.fit(train_x, train_y)

    # print results
    result = pd.DataFrame(grid_model_Ada.cv_results_)
    best_model_Ada = grid_model_Ada.best_estimator_

    #print("CV score")

    # DataFrame으로 만든 result중에 일부 컬럼만 출력함 
    result[["params"] + ["split" + str(i) + "_test_score" for i in range(5)] + ["std_test_score", "mean_test_score"]]

    """>>>### 검증"""

    if log_en==1:

        print("Adaboost")
        print("Best Parameter : " + str(grid_model_Ada.best_params_))

    # predict
    pred = best_model_Ada.predict(test_x)

    
    # pandas의 cross table  --> heatmap과 같아, accuracy 산정 활용
    tab = pd.crosstab(test_y, pred, rownames = ["real"], colnames = ["pred"])
    
    if log_en==1:
        print(tab)

    # Acc 계산/출력
    hit_count=0

    for i in range(pd.Series(pred).nunique()):
      hit_count = hit_count + tab.iloc[i,i]

    Acc_Ensemble =  hit_count / len(test_x)

    if log_en==1:

        print("Acc : " + str( Acc_Ensemble))

        # visualie the feature importance

        plt.figure(figsize = (10, 5))

        plt.title("Adaboost")
        fi = best_model_Ada.feature_importances_
        idx = (-fi).argsort()[0:10]
        fi = fi[idx]
        idx = train_x.columns[idx]
        plt.barh(range(10), fi[::-1], align='center')
        plt.xlim(0, 1)
        plt.yticks(range(10), idx[::-1])
        plt.xlabel('Feature importances', size=10)
        plt.ylabel('Feature', size=10)

        plt.show()

    """>>## **Ensemble(RandomForest)**"""

    from sklearn.ensemble import RandomForestClassifier   # RandomForestRegressor

    params = {"n_estimators" : [5, 10, 20]}

    """>>>### 분류기 모형 / 훈련"""

    #define model
    clf = RandomForestClassifier(max_depth=8,random_state=seed_no) # 모델 정의


    # model train with gridsearchCV    # CV = CrossValidation    5개의 데이터셑으로 분할하고, 4번 훈련, 1번 test 수행 
    grid_model_Bag = GridSearchCV(clf, param_grid = params, cv = 5, refit = True, return_train_score= True)

    #train model
    grid_model_Bag.fit(train_x, train_y)

    # print results
    result = pd.DataFrame(grid_model_Bag.cv_results_)
    best_model_Bag = grid_model_Bag.best_estimator_

    #print("CV score")

    # DataFrame으로 만든 result중에 일부 컬럼만 출력함 
    result[["params"] + ["split" + str(i) + "_test_score" for i in range(5)] + ["std_test_score", "mean_test_score"]]

    """>>>### 검증"""

    # predict
    pred = best_model_Bag.predict(test_x)

    # pandas의 cross table  --> heatmap과 같아, accuracy 산정 활용
    tab = pd.crosstab(test_y, pred, rownames = ["real"], colnames = ["pred"])

    if log_en==1:    
        print(tab)

    # Acc 계산/출력
    hit_count=0

    for i in range(pd.Series(pred).nunique()):
      hit_count = hit_count + tab.iloc[i,i]

    Acc_RForest =  hit_count / len(test_x)

    if log_en==1:

        print("Acc : " + str( Acc_RForest))

    """# **결과**
    ---
    >Data Augmentation 유/무, 모형별 Accuracy 비교
    """

    if log_en==1:
        """Data Augmentation 유/무, 모형별, Accuracy 비교"""
        print(f"DNN: {Acc_DNN:.3f}, MLP:{Acc_MLP:.3f}, DTree:{Acc_TREE:.3f}, KNN:{Acc_KNN:.3f}, AdaBoost: {Acc_Ensemble:.3f}, RandomForest: {Acc_RForest:.3f}")

    # No Data Augmentation
    # DNN: 0.872, MLP:0.842, DTree:0.744, KNN:0.789, AdaBoost: 0.556, RandomForest: 0.872

    # Data Augmentation
    # DNN: 0.882, MLP:0.882, DTree:0.729, KNN:0.857, AdaBoost: 0.490, RandomForest: 0.862  (jittering sigma = 0.05,  MagWarp = 0.2)
    # DNN: 0.977, MLP:0.967, DTree:0.854, KNN:0.960, AdaBoost: 0.686, RandomForest: 0.950  (jittering sigma = 0.005, MagWarp = 0.2)
    # DNN: 0.975, MLP:0.967, DTree:0.859, KNN:0.960, AdaBoost: 0.704, RandomForest: 0.975  (jittering sigma = 0.005, MagWarp = 0.05)
    # DNN: 0.992, MLP:0.987, DTree:0.900, KNN:0.985, AdaBoost: 0.666, RandomForest: 0.958  (jittering sigma = 0.005, MagWarp = 0.05, Scaling = 0.1)
    # DNN: 0.985, MLP:0.979, DTree:0.903, KNN:0.985, AdaBoost: 0.745, RandomForest: 0.937  (jittering sigma = 0.005, MagWarp = 0.05, Scaling = 0.025, Combination = (0.1,0.01))
    
    
    return Acc_DNN,Acc_MLP,Acc_TREE,Acc_KNN,Acc_Ensemble, Acc_RForest



