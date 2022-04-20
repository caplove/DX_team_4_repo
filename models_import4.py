#
# models_import4.py
# my_models_2(x,y,x_list,log_en,external_test_data=0,ext_x,ext_y)
# x, y는 원본데이터
# x_list는 [원본 x, 증강 x1, 증강 x2, ...]
# 모형은 1D-CNN, GRU /w Attention, Inception
#
#
# 1. 데이터 가공부 : x,y,x_list  --> train / valid / test용 로더 구축
# 2. 






def my_models_1D_CNN(x,y,x_list,log_en,external_test_data=0,ext_x=[],ext_y=[]):

    """1. 데이터 가공부"""
 
    # 모듈 불러오기
    import os
    import time
    import copy
    import random
    # import pickle       # 데이터 저장형태가 pickle일 경우 사용함
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim


    # Hyperparameter setting
    batch_size = 10
    num_classes = 16
    num_epochs = 200
    window_size = 50  # 22.03.21

    random_seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available

    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    """# 전처리(로더)

    ### 데이터 가공
    >> - 데이터를 window size 배수만큼 자르고, train / valid / test 로 나누기

    >> - 함수
    >>> * split_train_test (x,y,x_list,window_size) : x_list에 있는 증강 데이터를 train / valid / test로 8:1:1 split함. 
    >>> * chunk_merge ( splited_x, splited_y, window_size) : train / valid /test로 나눈 데이터를 class별 400개씩 자르고, 이를 merge함.
    >>>    - class별로 400개씩 자른 이유는 동일한 label을 갖도록 하기 위해서임.
    """

    # split_train_test

    """ 함수정의 """

    def split_train_test(x, y, x_list, window_size):
        import pandas as pd
        x_train_all=pd.DataFrame()
        x_valid_all=pd.DataFrame()
        x_test_all=pd.DataFrame()
        y_train_all=pd.DataFrame()
        y_valid_all=pd.DataFrame()
        y_test_all=pd.DataFrame()
        class_list=np.array(['idle', 'suit_1','suit_2','suit_3','suit_4','suit_5','shirt_1','shirt_2','shirt_3',
                'shirt_4','shirt_5','coat_1','coat_2','coat_3','coat_4','coat_5'])    
        for ii in x_list:
        # Augmented x data
            """window_size로 잘라 reshape가능하도록"""
            # 8:1:1 split
            for i in class_list:       # class별로 'idle', 'suit1', ....

                idx_class = y==i
                """클래스별 데이터셑 나누기"""
                # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
                # train, validation, test data의 개수 설정
                n_train = int(0.6 * len(ii[idx_class]))
                n_valid = int(0.2 * len(ii[idx_class]))
                n_test  = int(0.2 * len(ii[idx_class]))
                # print(n_train,n_valid,n_test)

                # train/validation set의 개수에 맞게 데이터 분할
                x_train, y_train = ii[idx_class][:n_train], y[idx_class][:n_train]
                x_valid, y_valid = ii[idx_class][n_train:n_train+n_valid], y[idx_class][n_train:n_train+n_valid]
                x_test,  y_test  = ii[idx_class][n_train+n_valid:], y[idx_class][n_train+n_valid:]
#                 """클래스별 데이터셑 나누기"""
#                 # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
#                 # train, validation, test data의 개수 설정
#                 n_train = int(0.8 * len(x[idx_class]))
#                 n_valid = int(0.1 * len(x[idx_class]))
#                 n_test  = int(0.1 * len(x[idx_class]))
#                 # print(n_train,n_valid,n_test)

#                 # train/validation set의 개수에 맞게 데이터 분할
#                 x_train, y_train = x[idx_class][:n_train], y[idx_class][:n_train]
#                 x_valid, y_valid = x[idx_class][n_train:n_train+n_valid], y[idx_class][n_train:n_train+n_valid]
#                 x_test,  y_test  = x[idx_class][n_train+n_valid:], y[idx_class][n_train+n_valid:]

                # print(f"클래스:{i},학습 : {x_train.shape}{y_train.shape}, 검증 : {x_valid.shape}, 테스트 : {x_test.shape}")
                """class별로 merge"""
                x_train_all = pd.concat([x_train_all,x_train],axis=0)
                y_train_all = pd.concat([y_train_all,y_train])
                x_valid_all = pd.concat([x_valid_all,x_valid],axis=0)
                y_valid_all = pd.concat([y_valid_all,y_valid])
                x_test_all  = pd.concat([x_test_all,x_test],axis=0)
                y_test_all  = pd.concat([y_test_all,y_test])
        return x_train_all,y_train_all,x_valid_all,y_valid_all,x_test_all,y_test_all



    # list 들어오면, n개씩 쪼개서 return   https://jsikim1.tistory.com/141
    def list_chunk(lst,n):
        return [lst[i:i+n] for i in range(0,len(lst),n)], len(lst)//n      #  // 몫


    def chunk_merge(x, y, no_of_data):
        class_list=np.array(['idle', 'suit_1','suit_2','suit_3','suit_4','suit_5','shirt_1','shirt_2','shirt_3',
                'shirt_4','shirt_5','coat_1','coat_2','coat_3','coat_4','coat_5'])
        df_x = pd.DataFrame()
        df_y = pd.DataFrame()
        xy=pd.concat([x,y],axis=1)
        xy.columns = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','current','class']    
        # chunk
        for i in class_list:       # class별로 'idle', 'suit1', ....
            df_all, count = list_chunk(xy[xy['class'] ==i],no_of_data)
        df_xy = pd.DataFrame(xy)
        df_x = np.array(df_xy.drop('class',axis=1))
        df_y = np.array(df_xy['class'])

        """class를 숫자로 변환 for pytorch"""
        y_number=[]
        for i in df_y:
            y_tmp=(np.where(i == class_list))[0][0]    # [0][0] 추가해서 데이터만 추출
            # print (i,y_tmp)
            y_number.append(y_tmp)
            # print(i,y_tmp[0][0])
        df_y=np.array(y_number)

        return df_x, df_y
   
   
    
    
    
    
    """데이터 list 정의"""
    # 함수의 입력인자로 받았으므로, 주석처리
    # x_list = [x]
    # x_list = [x, x_in_jittering, x_in_MagWarp, x_in_Scaling, x_in_Combination]

    """torch용 data 생성"""
    # no_of_data = 400
    # window_size = 400 
    # 데이터를 8:1:1로 나누고
    x_train_all, y_train_all, x_valid_all, y_valid_all, x_test_all, y_test_all = split_train_test (x, y, x_list, window_size)
    # class별로 400개씩 나누어 떨어지도록 나눔.
    x_train, y_train = chunk_merge(x_train_all,y_train_all,window_size)
    x_valid, y_valid = chunk_merge(x_valid_all,y_valid_all,window_size)
    x_test, y_test   = chunk_merge(x_test_all, y_test_all, window_size)

    # print(x_test.shape,y_test.shape,x_train.shape,y_train.shape)
    print(x_train_all.shape, y_train_all.shape, x_valid_all.shape, y_valid_all.shape, x_test_all.shape, y_test_all.shape)

    """### 데이터 변환
    > - torch용 로더 입력용 변환
    >> 2차원 데이터( #####, 7) --> 3차원데이터 (###, 7, 400)로 변환
    >>> 7은 센서의 개수, 400은 데이터 싯점,윈도우,4초의 표현임
    """

    # train/validation/test 데이터를 window_size 시점 길이로 분할
    datasets = []

    for set in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        # 전체 시간 길이 설정
        T = set[0].shape[0]

        # 전체 X 데이터를 window_size 크기의 time window로 분할
        # split(array, indices_or_sections) 함수는 자투리 없이 딱 나누어 떨어져야 하므로, 400으로 나눠떨어지도록 자투리 처리, split은 딱 떨어져야 함..
        # array 부분을   set[0].iloc[:window_size * (T // window_size),:] 로 slicing 먼저해주어야 함.
        # windows = np.split(set[0].iloc[:window_size * (T // window_size),:], T // window_size, axis=0)  


        x_sliced = set[0][:window_size * (T // window_size),:]
        x_sliced_transposed = x_sliced.T
        windows = np.split(x_sliced_transposed,T // window_size, axis=1)

        # split 하고난, windows는 list형태로 돌아가므로 다시 array 형태로 변환해야 함.
        windows = np.concatenate(windows, axis=0) # 세로로 이어붙임.
        if log_en:
            print("windows_original:",windows.shape)
        # print(windows[:7,:])
        # windows = windows.reshape(window_size,7,-1)
        windows = windows.reshape(-1,7,window_size)

        if log_en:
            print("windows_reshaped:",windows.shape)
        # print(windows[0,:,:])

        # 전체 y 데이터를 window_size 크기에 맞게 분할
        # labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1) # y는 2차원이므로...
        # labels = np.round(np.mean(np.concatenate(labels, 0), -1))  # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

        labels = np.split(set[1][:window_size * (T // window_size)], T // window_size) # y는 2차원이므로...
        labels = np.round(np.mean((np.concatenate(labels, 0).reshape(-1,window_size)), -1))  
        # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

        labels = labels.astype(np.long)
        # print(labels[0])

        # shape 확인
        if log_en:
            print(windows.shape,labels.shape)


        # sample data 확인 (마지막 15번 데이터)
        if log_en:
            print("sample data : last label")
            print(windows[-1],labels[-1])
           
            
        # 분할된 time window 단위의 X, y 데이터를 tensor 형태로 축적
        datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))
        
    # train/validation/test DataLoader 구축
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    

    
    
    
    
    
    
    
    
    """ 1D CNN"""

    ## 모형 설계

    KERNEL_SIZE_PARAM = 10

    # 1-dimensional convolution layer로 구성된 CNN 모델
    # 2개의 1-dimensional convolution layer와 1개의 fully-connected layer로 구성되어 있음
    class CNN_1D(nn.Module):
        def __init__(self, num_classes):
            super(CNN_1D, self).__init__()
            # 첫 번째 1-dimensional convolution layer 구축
            self.layer1 = nn.Sequential(                # Conv, Relu, Avg를 한번에 레이어로 구성하고, 이를 Sequencial로 사용함.
                # nn.Conv1d(7, 7, kernel_size = KERNEL_SIZE_PARAM),      # Conv(input(입력피쳐개수),output(필터의 개수(종류)), filter_size(필터의 크기 시간축으로 몇개 볼것인지))
                nn.Conv1d(7, 32, kernel_size=KERNEL_SIZE_PARAM),      # Conv(input,output, 3개를 한번에 보겠다.)
                # nn.Conv1d(561, 64, kernel_size=3),      # Conv(input,output, 3개를 한번에 보겠다.)
                nn.ReLU(),
                nn.AvgPool1d(2)                         # 
            )
            # 두 번째 1-dimensional convolution layer 구축
            self.layer2 = nn.Sequential(
                nn.Conv1d(32,32, kernel_size = KERNEL_SIZE_PARAM),
                nn.ReLU(),
                nn.AvgPool1d(2)
            )
            # fully-connected layer 구축
            # self.fc = nn.Linear(64 * 11, num_classes)
            # self.fc = nn.Linear(64 * 1, num_classes)
            self.fc = nn.Linear(160, num_classes)            # 아래 훈련단계에서 에러날경우, nn.Liner(0000, num_classes)  0000 을 에러로그를 보고 바꾼다.(주의)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    # 1D CNN 구축
    model_CNN = CNN_1D(num_classes=num_classes)
    model_CNN = model_CNN.to(device)
    if log_en:
        print(model_CNN)

    # SGD optimizer 구축하기
    optimizer = optim.SGD(model_CNN.parameters(), lr=0.001, momentum=0.9)

    def train_model_CNN(model, dataloaders, criterion, num_epochs, optimizer):
        since = time.time()

        val_acc_history_CNN = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):

            if epoch % 50==0 and log_en:
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print('-' * 10)

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_corrects = 0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.long)
                    # labels = labels.to(device, dtype=torch.int64)

                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                        _, preds = torch.max(outputs, 1)

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += labels.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total
                epoch_acc = running_corrects.double() / running_total

                if epoch % 50==0 and log_en:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_acc > best_acc:

                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history_CNN.append(epoch_acc)


        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        if log_en:
            print('1D_CNN Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)

        # best model 가중치 저장
        # torch.save(best_model_wts, '../output/best_model.pt')
        return model, val_acc_history_CNN

    
    
    
    # training 단계에서 사용할 Dataloader dictionary 생성
    dataloaders_dict = {
        'train': train_loader,
        'val': valid_loader
    }

    # loss function 설정
    criterion = nn.CrossEntropyLoss()

    """## 학습 / 검증"""
    # 모델 학습  --> 모형, accuracy array 리턴
    model_CNN, val_acc_history_CNN = train_model_CNN (model_CNN, dataloaders_dict, criterion, num_epochs, optimizer)

    """## 테스트"""
    def test_model_CNN(model, test_loader):
        model.eval()   # 모델을 validation mode로 설정

        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            corrects = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)

                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)

                # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                _, preds = torch.max(outputs, 1)

                # batch별 정답 개수를 축적함
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        # accuracy를 도출함
        test_acc = corrects.double() / total
        if log_en:
            print('1D_CNN Testing Acc: {:.4f}'.format(test_acc))
        
        return test_acc
        

    # 모델 검증 --> Accuracy 리턴
    Acc_1D_CNN_valid = test_model_CNN (model_CNN, test_loader)

 
    """## 별도의 Test Data로 검증"""
    Acc_1D_CNN_external_data = 0
    if external_test_data:

        x_list = [ext_x]   # 별도의 test 에서는 augment 하지 않으므로, x_list는 원본 external x만 사용한다.

        x_train_all,y_train_all,x_valid_all,y_valid_all,x_test_all, y_test_all = split_train_test(ext_x, ext_y, x_list, window_size)

        """torch용 data 생성"""
        # no_of_data=400

        x_train, y_train = chunk_merge(x_train_all,y_train_all,window_size)
        x_valid, y_valid = chunk_merge(x_valid_all,y_valid_all,window_size)
        x_test, y_test   = chunk_merge(x_test_all, y_test_all, window_size)

        # train/validation/test 데이터를 window_size 시점 길이로 분할
        datasets = []
        for set in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
            # 전체 시간 길이 설정
            T = set[0].shape[0]

            # 전체 X 데이터를 window_size 크기의 time window로 분할
            # split(array, indices_or_sections) 함수는 자투리 없이 딱 나누어 떨어져야 하므로, 400으로 나눠떨어지도록 자투리 처리, split은 딱 떨어져야 함..
            # array 부분을   set[0].iloc[:window_size * (T // window_size),:] 로 slicing 먼저해주어야 함.
            # windows = np.split(set[0].iloc[:window_size * (T // window_size),:], T // window_size, axis=0)  


            x_sliced = set[0][:window_size * (T // window_size),:]
            x_sliced_transposed = x_sliced.T
            windows = np.split(x_sliced_transposed,T // window_size, axis=1)

            # split 하고난, windows는 list형태로 돌아가므로 다시 array 형태로 변환해야 함.
            windows = np.concatenate(windows, axis=0) # 세로로 이어붙임.
            if log_en:
                print("windows_original:",windows.shape)
            # print(windows[:7,:])
            # windows = windows.reshape(window_size,7,-1)
            windows = windows.reshape(-1,7,window_size)

            if log_en:
                print("windows_reshaped:",windows.shape)
            # print(windows[0,:,:])

            # 전체 y 데이터를 window_size 크기에 맞게 분할
            # labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1) # y는 2차원이므로...
            # labels = np.round(np.mean(np.concatenate(labels, 0), -1))  # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

            labels = np.split(set[1][:window_size * (T // window_size)], T // window_size) # y는 2차원이므로...
            labels = np.round(np.mean((np.concatenate(labels, 0).reshape(-1,window_size)), -1))  
            # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

            labels = labels.astype(np.long)
            # print(labels[0])

            # shape 확인
            if log_en:
                print(windows.shape,labels.shape)


            # sample data 확인 (마지막 15번 데이터)
            if log_en:
                print("sample data : last label")
                print(windows[-1],labels[-1])

            # 분할된 time window 단위의 X, y 데이터를 tensor 형태로 축적
            datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))


        # train/validation/test DataLoader 구축
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # 모델 검증 (새로운 싯점 데이터 기준 )
        Acc_1D_CNN_external_data = test_model_CNN (model_CNN, train_loader)   # 별도의 데이터는 train_loader를 상용한다.

    print("STATUS : 1D_CNN 모형 완료")

    return np.max(val_acc_history_CNN), Acc_CNN_valid, Acc_CNN_external_data
    
    
    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
    
    
    
    
    
    
    
    
    
    
    
    
    
   def my_models_GRU(x,y,x_list,log_en,external_test_data=0,ext_x=[],ext_y=[]):

    """1. 데이터 가공부"""
 
    # 모듈 불러오기
    import os
    import time
    import copy
    import random
    # import pickle       # 데이터 저장형태가 pickle일 경우 사용함
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim


    # Hyperparameter setting
    batch_size = 10
    num_classes = 16
    num_epochs = 200
    window_size = 50  # 22.03.21

    random_seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available

    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    """# 전처리(로더)

    ### 데이터 가공
    >> - 데이터를 window size 배수만큼 자르고, train / valid / test 로 나누기

    >> - 함수
    >>> * split_train_test (x,y,x_list,window_size) : x_list에 있는 증강 데이터를 train / valid / test로 8:1:1 split함. 
    >>> * chunk_merge ( splited_x, splited_y, window_size) : train / valid /test로 나눈 데이터를 class별 400개씩 자르고, 이를 merge함.
    >>>    - class별로 400개씩 자른 이유는 동일한 label을 갖도록 하기 위해서임.
    """

    # split_train_test

    """ 함수정의 """

    def split_train_test(x, y, x_list, window_size):
        import pandas as pd
        x_train_all=pd.DataFrame()
        x_valid_all=pd.DataFrame()
        x_test_all=pd.DataFrame()
        y_train_all=pd.DataFrame()
        y_valid_all=pd.DataFrame()
        y_test_all=pd.DataFrame()
        class_list=np.array(['idle', 'suit_1','suit_2','suit_3','suit_4','suit_5','shirt_1','shirt_2','shirt_3',
                'shirt_4','shirt_5','coat_1','coat_2','coat_3','coat_4','coat_5'])    
        for ii in x_list:
        # Augmented x data
            """window_size로 잘라 reshape가능하도록"""
            # 8:1:1 split
            for i in class_list:       # class별로 'idle', 'suit1', ....

                idx_class = y==i
                """클래스별 데이터셑 나누기"""
                # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
                # train, validation, test data의 개수 설정
                n_train = int(0.6 * len(ii[idx_class]))
                n_valid = int(0.2 * len(ii[idx_class]))
                n_test  = int(0.2 * len(ii[idx_class]))
                # print(n_train,n_valid,n_test)

                # train/validation set의 개수에 맞게 데이터 분할
                x_train, y_train = ii[idx_class][:n_train], y[idx_class][:n_train]
                x_valid, y_valid = ii[idx_class][n_train:n_train+n_valid], y[idx_class][n_train:n_train+n_valid]
                x_test,  y_test  = ii[idx_class][n_train+n_valid:], y[idx_class][n_train+n_valid:]
#                 """클래스별 데이터셑 나누기"""
#                 # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
#                 # train, validation, test data의 개수 설정
#                 n_train = int(0.8 * len(x[idx_class]))
#                 n_valid = int(0.1 * len(x[idx_class]))
#                 n_test  = int(0.1 * len(x[idx_class]))
#                 # print(n_train,n_valid,n_test)

#                 # train/validation set의 개수에 맞게 데이터 분할
#                 x_train, y_train = x[idx_class][:n_train], y[idx_class][:n_train]
#                 x_valid, y_valid = x[idx_class][n_train:n_train+n_valid], y[idx_class][n_train:n_train+n_valid]
#                 x_test,  y_test  = x[idx_class][n_train+n_valid:], y[idx_class][n_train+n_valid:]

                # print(f"클래스:{i},학습 : {x_train.shape}{y_train.shape}, 검증 : {x_valid.shape}, 테스트 : {x_test.shape}")
                """class별로 merge"""
                x_train_all = pd.concat([x_train_all,x_train],axis=0)
                y_train_all = pd.concat([y_train_all,y_train])
                x_valid_all = pd.concat([x_valid_all,x_valid],axis=0)
                y_valid_all = pd.concat([y_valid_all,y_valid])
                x_test_all  = pd.concat([x_test_all,x_test],axis=0)
                y_test_all  = pd.concat([y_test_all,y_test])
        return x_train_all,y_train_all,x_valid_all,y_valid_all,x_test_all,y_test_all



    # list 들어오면, n개씩 쪼개서 return   https://jsikim1.tistory.com/141
    def list_chunk(lst,n):
        return [lst[i:i+n] for i in range(0,len(lst),n)], len(lst)//n      #  // 몫


    def chunk_merge(x, y, no_of_data):
        class_list=np.array(['idle', 'suit_1','suit_2','suit_3','suit_4','suit_5','shirt_1','shirt_2','shirt_3',
                'shirt_4','shirt_5','coat_1','coat_2','coat_3','coat_4','coat_5'])
        df_x = pd.DataFrame()
        df_y = pd.DataFrame()
        xy=pd.concat([x,y],axis=1)
        xy.columns = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','current','class']    
        # chunk
        for i in class_list:       # class별로 'idle', 'suit1', ....
            df_all, count = list_chunk(xy[xy['class'] ==i],no_of_data)
        df_xy = pd.DataFrame(xy)
        df_x = np.array(df_xy.drop('class',axis=1))
        df_y = np.array(df_xy['class'])

        """class를 숫자로 변환 for pytorch"""
        y_number=[]
        for i in df_y:
            y_tmp=(np.where(i == class_list))[0][0]    # [0][0] 추가해서 데이터만 추출
            # print (i,y_tmp)
            y_number.append(y_tmp)
            # print(i,y_tmp[0][0])
        df_y=np.array(y_number)

        return df_x, df_y
   
   
    
    
    
    
    """데이터 list 정의"""
    # 함수의 입력인자로 받았으므로, 주석처리
    # x_list = [x]
    # x_list = [x, x_in_jittering, x_in_MagWarp, x_in_Scaling, x_in_Combination]

    """torch용 data 생성"""
    # no_of_data = 400
    # window_size = 400 
    # 데이터를 8:1:1로 나누고
    x_train_all, y_train_all, x_valid_all, y_valid_all, x_test_all, y_test_all = split_train_test (x, y, x_list, window_size)
    # class별로 400개씩 나누어 떨어지도록 나눔.
    x_train, y_train = chunk_merge(x_train_all,y_train_all,window_size)
    x_valid, y_valid = chunk_merge(x_valid_all,y_valid_all,window_size)
    x_test, y_test   = chunk_merge(x_test_all, y_test_all, window_size)

    # print(x_test.shape,y_test.shape,x_train.shape,y_train.shape)
    print(x_train_all.shape, y_train_all.shape, x_valid_all.shape, y_valid_all.shape, x_test_all.shape, y_test_all.shape)

    """### 데이터 변환
    > - torch용 로더 입력용 변환
    >> 2차원 데이터( #####, 7) --> 3차원데이터 (###, 7, 400)로 변환
    >>> 7은 센서의 개수, 400은 데이터 싯점,윈도우,4초의 표현임
    """

    # train/validation/test 데이터를 window_size 시점 길이로 분할
    datasets = []

    for set in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        # 전체 시간 길이 설정
        T = set[0].shape[0]

        # 전체 X 데이터를 window_size 크기의 time window로 분할
        # split(array, indices_or_sections) 함수는 자투리 없이 딱 나누어 떨어져야 하므로, 400으로 나눠떨어지도록 자투리 처리, split은 딱 떨어져야 함..
        # array 부분을   set[0].iloc[:window_size * (T // window_size),:] 로 slicing 먼저해주어야 함.
        # windows = np.split(set[0].iloc[:window_size * (T // window_size),:], T // window_size, axis=0)  


        x_sliced = set[0][:window_size * (T // window_size),:]
        x_sliced_transposed = x_sliced.T
        windows = np.split(x_sliced_transposed,T // window_size, axis=1)

        # split 하고난, windows는 list형태로 돌아가므로 다시 array 형태로 변환해야 함.
        windows = np.concatenate(windows, axis=0) # 세로로 이어붙임.
        if log_en:
            print("windows_original:",windows.shape)
        # print(windows[:7,:])
        # windows = windows.reshape(window_size,7,-1)
        windows = windows.reshape(-1,7,window_size)

        if log_en:
            print("windows_reshaped:",windows.shape)
        # print(windows[0,:,:])

        # 전체 y 데이터를 window_size 크기에 맞게 분할
        # labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1) # y는 2차원이므로...
        # labels = np.round(np.mean(np.concatenate(labels, 0), -1))  # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

        labels = np.split(set[1][:window_size * (T // window_size)], T // window_size) # y는 2차원이므로...
        labels = np.round(np.mean((np.concatenate(labels, 0).reshape(-1,window_size)), -1))  
        # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

        labels = labels.astype(np.long)
        # print(labels[0])

        # shape 확인
        if log_en:
            print(windows.shape,labels.shape)


        # sample data 확인 (마지막 15번 데이터)
        if log_en:
            print("sample data : last label")
            print(windows[-1],labels[-1])
           
            
        # 분할된 time window 단위의 X, y 데이터를 tensor 형태로 축적
        datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))
        
    # train/validation/test DataLoader 구축
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False) 
    
    
    
    
    
    
    
    
    
    """# GRU /w Attention"""

    # 모듈 불러오기
    import os
    import time
    import copy
    import random
    # import pickle       # 데이터 저장형태가 pickle일 경우 사용함
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    # Hyperparameter setting
    batch_size = 32
    num_classes = 16
    num_epochs = 50
    # window_size = 400  # 몇 시점의 데이터를 넣을것인가.
    input_size = 7     # 7개의 변수  (7차원)
    hidden_size = 64    # hidden layer의 차원은 (64차원)
    num_layers = 2
    bidirectional = True

    random_seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available

    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    """## 모형 설계

    """

    class Attention(nn.Module):
        def __init__(self, device, hidden_size):
            super(Attention, self).__init__()
            self.device = device
            self.hidden_size = hidden_size

            self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)

        def forward(self, rnn_outputs, final_hidden_state):        # 입력은 (h1 ~ h*, h* )
            # rnn_output.shape:         (batch_size, seq_len, hidden_size)
            # final_hidden_state.shape: (batch_size, hidden_size)
            # NOTE: hidden_size may also reflect bidirectional hidden states (hidden_size = num_directions * hidden_dim)
            batch_size, seq_len, _ = rnn_outputs.shape  # bidirect에서는 batch 2배

            attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
            attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))   # unsqueeze(2)는 두번째 축을 만듬. 2차원 --> 3차원으로 (batch,hidden,1)

            # bmm : 돌려서 행렬곱 시행함.   ---> 결과 alpha1, ~ alpha50까지 나옴.


            attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)

            # alpha1*h1 + alpha2*h2 + ... alpha50*h50
            context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)

            # Wc 가 concat_linear 
            attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))

            return attn_hidden, attn_weights

    # GRU_Attention

    class GRU_Attention(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional, device):
            super(GRU_Attention, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_directions = 2 if bidirectional == True else 1

            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.attn = Attention(device, hidden_size * self.num_directions)   # 차이점 : Attention함수로 context vector를 도출하고 이결과가 fc로 들어감.
            self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

        def forward(self, x):
            batch_size, _, seq_len = x.shape

            # data dimension: (batch_size x input_size x seq_len) -> (batch_size x seq_len x input_size)로 변환
            x = torch.transpose(x, 1, 2)

            # initial hidden states 설정
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

            # out: tensor of shape (batch_size, seq_len, hidden_size)
            rnn_output, hiddens = self.rnn(x, h0)    # 출력 둘중의 하나를 가지고 atten을 하게되는데...
            final_state = hiddens.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)[-1]
                          # view는 shape을 변경하는 method임.                                             # -1은 num_layers중에서 마지막것을 가져오려고...    
                          # bydirectional 구조에서 마지막 히든레이어의 정방향, 역방향 중 정방향 h를 가져온것 

            # Handle directions
            final_hidden_state = None
            if self.num_directions == 1:   # 단뱡향
                final_hidden_state = final_state.squeeze(0) # 0번째 축(차원)을 없앰. 
            elif self.num_directions == 2: # 양방향
                h_1, h_2 = final_state[0], final_state[1]
                final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states    64차원 2개를 이어붙여서 128차원으로 concat 함.

            # final hidden state 기준으로 a 값들과의 상관도를 계산함.

            # Push through attention layer
            attn_output, attn_weights = self.attn(rnn_output, final_hidden_state)

            attn_output = self.fc(attn_output)
            return attn_output

    # GRU 모델 구축
    gru = GRU_Attention(input_size, hidden_size, num_layers, num_classes, bidirectional, device)
    gru = gru.to(device)
    if log_en:
        print(gru)

    """## 학습"""

    def train_model_GRU(model, dataloaders, criterion, num_epochs, optimizer):
        since = time.time()

        val_acc_history_GRU = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):


            if epoch % 20==0 and log_en:
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print('-' * 10)

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_corrects = 0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.long)

                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                        _, preds = torch.max(outputs, 1)

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += labels.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total
                epoch_acc = running_corrects.double() / running_total

                if epoch % 20==0 and log_en:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history_GRU.append(epoch_acc)

            # print()

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        if log_en:
            print('GRU Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)

        # best model 가중치 저장
        # torch.save(best_model_wts, '../output/best_model.pt')
        return model, val_acc_history_GRU

    # trining 단계에서 사용할 Dataloader dictionary 생성
    dataloaders_dict = {
        'train': train_loader,
        'val': valid_loader
    }

    # loss function 설정
    criterion = nn.CrossEntropyLoss()

    # GRU with attention 모델 학습
    gru, val_acc_history_GRU = train_model_GRU(gru, dataloaders_dict, criterion, num_epochs,
                                           optimizer=optim.Adam(gru.parameters(), lr=0.001))

    """## 검증"""

    def test_model_GRU(model, test_loader):
        model.eval()   # 모델을 validation mode로 설정

        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            corrects = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)

                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)

                # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                _, preds = torch.max(outputs, 1)

                # batch별 정답 개수를 축적함
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        # accuracy를 도출함
        test_acc = corrects.double() / total
        if log_en:
            print('GRU Testing Acc: {:.4f}'.format(test_acc))
        
        
        return test_acc
        
        

    # GRU with attention 모델 검증하기 (Acc: 0.8889)
    # Benchmark model인 GRU(Acc: 0.8000)와 비교했을 때, Attetion의 적용이 성능 향상에 도움이 됨을 알 수 있음
    Acc_GRU_valid = test_model_GRU(gru, test_loader)

    """## 별도의 Test Data로 검증"""
    Acc_GRU_external_data=0
    if external_test_data:
        x_list = [ext_x]

        x_train_all,y_train_all,x_valid_all,y_valid_all,x_test_all,y_test_all = split_train_test(ext_x, ext_y, x_list, window_size)

        """torch용 data 생성"""
        # no_of_data=400

        x_train, y_train = chunk_merge(x_train_all,y_train_all,window_size)
        x_valid, y_valid = chunk_merge(x_valid_all,y_valid_all,window_size)
        x_test, y_test   = chunk_merge(x_test_all, y_test_all, window_size)

        # train/validation/test 데이터를 window_size 시점 길이로 분할
        datasets = []
        for set in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
            # 전체 시간 길이 설정
            T = set[0].shape[0]

            # 전체 X 데이터를 window_size 크기의 time window로 분할
            # split(array, indices_or_sections) 함수는 자투리 없이 딱 나누어 떨어져야 하므로, 400으로 나눠떨어지도록 자투리 처리, split은 딱 떨어져야 함..
            # array 부분을   set[0].iloc[:window_size * (T // window_size),:] 로 slicing 먼저해주어야 함.
            # windows = np.split(set[0].iloc[:window_size * (T // window_size),:], T // window_size, axis=0)  


            x_sliced = set[0][:window_size * (T // window_size),:]
            x_sliced_transposed = x_sliced.T
            windows = np.split(x_sliced_transposed,T // window_size, axis=1) # axis=1 이면, 가로로 자름.

            # split 하고난, windows는 list형태로 돌아가므로 다시 array 형태로 변환해야 함.
            windows = np.concatenate(windows, axis=0) # axis=0 이면, 세로로 이어붙임.
            if log_en:
                print("windows_original:",windows.shape)
            # print(windows[:7,:])
            # windows = windows.reshape(window_size,7,-1)
            windows = windows.reshape(-1,7,window_size)

            if log_en:
                print("windows_reshaped:",windows.shape)
            # print(windows[0,:,:])

            # 전체 y 데이터를 window_size 크기에 맞게 분할
            # labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1) # y는 2차원이므로...
            # labels = np.round(np.mean(np.concatenate(labels, 0), -1))  # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

            labels = np.split(set[1][:window_size * (T // window_size)], T // window_size) # y는 2차원이므로...
            labels = np.round(np.mean((np.concatenate(labels, 0).reshape(-1,window_size)), -1))  # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

            labels = labels.astype(np.long)
            # print(labels[0])

            # shape 확인
            if log_en:
                print(windows.shape,labels.shape)


            # sample data 확인 (마지막 15번 데이터)
            if log_en:
                print("sample data : last label")
                print(windows[-1],labels[-1])

            # 분할된 time window 단위의 X, y 데이터를 tensor 형태로 축적
            datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))



        # train/validation/test DataLoader 구축
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # 모델 검증 (새로운 싯점 데이터 기준 )
        Acc_GRU_external_data = test_model_GRU(gru, train_loader)

    print("STATUS : GRU 완료")    
    
    return np.max(val_acc_history_GRU), Acc_GRU_valid, Acc_GRU_external_data
    
    
    
    
    
    
    
    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################    
###################################################################################################################      
    

    
    
    
    
    
    
    
    
    
def my_models_Incept(x,y,x_list,log_en,external_test_data=0,ext_x=[],ext_y=[]):

    """1. 데이터 가공부"""
 
    # 모듈 불러오기
    import os
    import time
    import copy
    import random
    # import pickle       # 데이터 저장형태가 pickle일 경우 사용함
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim


    # Hyperparameter setting
    batch_size = 10
    num_classes = 16
    num_epochs = 200
    window_size = 50  # 22.03.21

    random_seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available

    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    """# 전처리(로더)

    ### 데이터 가공
    >> - 데이터를 window size 배수만큼 자르고, train / valid / test 로 나누기

    >> - 함수
    >>> * split_train_test (x,y,x_list,window_size) : x_list에 있는 증강 데이터를 train / valid / test로 8:1:1 split함. 
    >>> * chunk_merge ( splited_x, splited_y, window_size) : train / valid /test로 나눈 데이터를 class별 400개씩 자르고, 이를 merge함.
    >>>    - class별로 400개씩 자른 이유는 동일한 label을 갖도록 하기 위해서임.
    """

    # split_train_test

    """ 함수정의 """

    def split_train_test(x, y, x_list, window_size):
        import pandas as pd
        x_train_all=pd.DataFrame()
        x_valid_all=pd.DataFrame()
        x_test_all=pd.DataFrame()
        y_train_all=pd.DataFrame()
        y_valid_all=pd.DataFrame()
        y_test_all=pd.DataFrame()
        class_list=np.array(['idle', 'suit_1','suit_2','suit_3','suit_4','suit_5','shirt_1','shirt_2','shirt_3',
                'shirt_4','shirt_5','coat_1','coat_2','coat_3','coat_4','coat_5'])    
        for ii in x_list:
        # Augmented x data
            """window_size로 잘라 reshape가능하도록"""
            # 8:1:1 split
            for i in class_list:       # class별로 'idle', 'suit1', ....

                idx_class = y==i
                """클래스별 데이터셑 나누기"""
                # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
                # train, validation, test data의 개수 설정
                n_train = int(0.6 * len(ii[idx_class]))
                n_valid = int(0.2 * len(ii[idx_class]))
                n_test  = int(0.2 * len(ii[idx_class]))
                # print(n_train,n_valid,n_test)

                # train/validation set의 개수에 맞게 데이터 분할
                x_train, y_train = ii[idx_class][:n_train], y[idx_class][:n_train]
                x_valid, y_valid = ii[idx_class][n_train:n_train+n_valid], y[idx_class][n_train:n_train+n_valid]
                x_test,  y_test  = ii[idx_class][n_train+n_valid:], y[idx_class][n_train+n_valid:]
#                 """클래스별 데이터셑 나누기"""
#                 # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
#                 # train, validation, test data의 개수 설정
#                 n_train = int(0.8 * len(x[idx_class]))
#                 n_valid = int(0.1 * len(x[idx_class]))
#                 n_test  = int(0.1 * len(x[idx_class]))
#                 # print(n_train,n_valid,n_test)

#                 # train/validation set의 개수에 맞게 데이터 분할
#                 x_train, y_train = x[idx_class][:n_train], y[idx_class][:n_train]
#                 x_valid, y_valid = x[idx_class][n_train:n_train+n_valid], y[idx_class][n_train:n_train+n_valid]
#                 x_test,  y_test  = x[idx_class][n_train+n_valid:], y[idx_class][n_train+n_valid:]

                # print(f"클래스:{i},학습 : {x_train.shape}{y_train.shape}, 검증 : {x_valid.shape}, 테스트 : {x_test.shape}")
                """class별로 merge"""
                x_train_all = pd.concat([x_train_all,x_train],axis=0)
                y_train_all = pd.concat([y_train_all,y_train])
                x_valid_all = pd.concat([x_valid_all,x_valid],axis=0)
                y_valid_all = pd.concat([y_valid_all,y_valid])
                x_test_all  = pd.concat([x_test_all,x_test],axis=0)
                y_test_all  = pd.concat([y_test_all,y_test])
        return x_train_all,y_train_all,x_valid_all,y_valid_all,x_test_all,y_test_all



    # list 들어오면, n개씩 쪼개서 return   https://jsikim1.tistory.com/141
    def list_chunk(lst,n):
        return [lst[i:i+n] for i in range(0,len(lst),n)], len(lst)//n      #  // 몫


    def chunk_merge(x, y, no_of_data):
        class_list=np.array(['idle', 'suit_1','suit_2','suit_3','suit_4','suit_5','shirt_1','shirt_2','shirt_3',
                'shirt_4','shirt_5','coat_1','coat_2','coat_3','coat_4','coat_5'])
        df_x = pd.DataFrame()
        df_y = pd.DataFrame()
        xy=pd.concat([x,y],axis=1)
        xy.columns = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','current','class']    
        # chunk
        for i in class_list:       # class별로 'idle', 'suit1', ....
            df_all, count = list_chunk(xy[xy['class'] ==i],no_of_data)
        df_xy = pd.DataFrame(xy)
        df_x = np.array(df_xy.drop('class',axis=1))
        df_y = np.array(df_xy['class'])

        """class를 숫자로 변환 for pytorch"""
        y_number=[]
        for i in df_y:
            y_tmp=(np.where(i == class_list))[0][0]    # [0][0] 추가해서 데이터만 추출
            # print (i,y_tmp)
            y_number.append(y_tmp)
            # print(i,y_tmp[0][0])
        df_y=np.array(y_number)

        return df_x, df_y
   
   
    
    
    
    
    """데이터 list 정의"""
    # 함수의 입력인자로 받았으므로, 주석처리
    # x_list = [x]
    # x_list = [x, x_in_jittering, x_in_MagWarp, x_in_Scaling, x_in_Combination]

    """torch용 data 생성"""
    # no_of_data = 400
    # window_size = 400 
    # 데이터를 8:1:1로 나누고
    x_train_all, y_train_all, x_valid_all, y_valid_all, x_test_all, y_test_all = split_train_test (x, y, x_list, window_size)
    # class별로 400개씩 나누어 떨어지도록 나눔.
    x_train, y_train = chunk_merge(x_train_all,y_train_all,window_size)
    x_valid, y_valid = chunk_merge(x_valid_all,y_valid_all,window_size)
    x_test, y_test   = chunk_merge(x_test_all, y_test_all, window_size)

    # print(x_test.shape,y_test.shape,x_train.shape,y_train.shape)
    print(x_train_all.shape, y_train_all.shape, x_valid_all.shape, y_valid_all.shape, x_test_all.shape, y_test_all.shape)

    """### 데이터 변환
    > - torch용 로더 입력용 변환
    >> 2차원 데이터( #####, 7) --> 3차원데이터 (###, 7, 400)로 변환
    >>> 7은 센서의 개수, 400은 데이터 싯점,윈도우,4초의 표현임
    """

    # train/validation/test 데이터를 window_size 시점 길이로 분할
    datasets = []

    for set in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        # 전체 시간 길이 설정
        T = set[0].shape[0]

        # 전체 X 데이터를 window_size 크기의 time window로 분할
        # split(array, indices_or_sections) 함수는 자투리 없이 딱 나누어 떨어져야 하므로, 400으로 나눠떨어지도록 자투리 처리, split은 딱 떨어져야 함..
        # array 부분을   set[0].iloc[:window_size * (T // window_size),:] 로 slicing 먼저해주어야 함.
        # windows = np.split(set[0].iloc[:window_size * (T // window_size),:], T // window_size, axis=0)  


        x_sliced = set[0][:window_size * (T // window_size),:]
        x_sliced_transposed = x_sliced.T
        windows = np.split(x_sliced_transposed,T // window_size, axis=1)

        # split 하고난, windows는 list형태로 돌아가므로 다시 array 형태로 변환해야 함.
        windows = np.concatenate(windows, axis=0) # 세로로 이어붙임.
        if log_en:
            print("windows_original:",windows.shape)
        # print(windows[:7,:])
        # windows = windows.reshape(window_size,7,-1)
        windows = windows.reshape(-1,7,window_size)

        if log_en:
            print("windows_reshaped:",windows.shape)
        # print(windows[0,:,:])

        # 전체 y 데이터를 window_size 크기에 맞게 분할
        # labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1) # y는 2차원이므로...
        # labels = np.round(np.mean(np.concatenate(labels, 0), -1))  # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

        labels = np.split(set[1][:window_size * (T // window_size)], T // window_size) # y는 2차원이므로...
        labels = np.round(np.mean((np.concatenate(labels, 0).reshape(-1,window_size)), -1))  
        # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

        labels = labels.astype(np.long)
        # print(labels[0])

        # shape 확인
        if log_en:
            print(windows.shape,labels.shape)


        # sample data 확인 (마지막 15번 데이터)
        if log_en:
            print("sample data : last label")
            print(windows[-1],labels[-1])
           
            
        # 분할된 time window 단위의 X, y 데이터를 tensor 형태로 축적
        datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))
        
    # train/validation/test DataLoader 구축
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)    

    
    
    
    
    
    """# InceptionTime

    > ## 모형 설계
    """

    import numpy as np 
    import time

    import torch 
    import torch.nn as nn
    import torch.nn.functional as F 

    import matplotlib.pyplot as plt
    from collections import OrderedDict

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    from sklearn.preprocessing import RobustScaler

    # Hyperparameter setting
    batch_size = 32
    num_classes = 16
    num_epochs = 400
    window_size = 400  # 몇 시점의 데이터를 넣을것인가.
    input_size = 7     # 7개의 변수  (7차원)
    hidden_size = 64    # hidden layer의 차원은 (64차원)

    random_seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available

    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    def correct_sizes(sizes):
        corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
        return corrected_sizes


    def pass_through(X):
        return X


    class Inception(nn.Module):
        def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
            """
            : param in_channels				Number of input channels (input features)
            : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
            : param kernel_sizes			List of kernel sizes for each convolution.
                                            Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                            This is nessesery because of padding size.
                                            For correction of kernel_sizes use function "correct_sizes". 
            : param bottleneck_channels		Number of output channels in bottleneck. 
                                            Bottleneck wont be used if nuber of in_channels is equal to 1.
            : param activation				Activation function for output tensor (nn.ReLU()). 
            : param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
            """
            super(Inception, self).__init__()
            self.return_indices=return_indices
            if in_channels > 1:
                self.bottleneck = nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=bottleneck_channels, 
                                    kernel_size=1, 
                                    stride=1, 
                                    bias=False
                                    )
            else:
                self.bottleneck = pass_through
                bottleneck_channels = 1

            self.conv_from_bottleneck_1 = nn.Conv1d(
                                            in_channels=bottleneck_channels, 
                                            out_channels=n_filters, 
                                            kernel_size=kernel_sizes[0], 
                                            stride=1, 
                                            padding=kernel_sizes[0]//2, 
                                            bias=False
                                            )
            self.conv_from_bottleneck_2 = nn.Conv1d(
                                            in_channels=bottleneck_channels, 
                                            out_channels=n_filters, 
                                            kernel_size=kernel_sizes[1], 
                                            stride=1, 
                                            padding=kernel_sizes[1]//2, 
                                            bias=False
                                            )
            self.conv_from_bottleneck_3 = nn.Conv1d(
                                            in_channels=bottleneck_channels, 
                                            out_channels=n_filters, 
                                            kernel_size=kernel_sizes[2], 
                                            stride=1, 
                                            padding=kernel_sizes[2]//2, 
                                            bias=False
                                            )
            self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
            self.conv_from_maxpool = nn.Conv1d(
                                        in_channels=in_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=1, 
                                        stride=1,
                                        padding=0, 
                                        bias=False
                                        )
            self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
            self.activation = activation

        def forward(self, X):
            # step 1
            Z_bottleneck = self.bottleneck(X)
            if self.return_indices:
                Z_maxpool, indices = self.max_pool(X)
            else:
                Z_maxpool = self.max_pool(X)
            # step 2
            Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
            Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
            Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
            Z4 = self.conv_from_maxpool(Z_maxpool)
            # step 3 
            Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
            Z = self.activation(self.batch_norm(Z))
            if self.return_indices:
                return Z, indices
            else:
                return Z


    class InceptionBlock(nn.Module):
        def __init__(self, in_channels, n_filters=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), return_indices=False):
            super(InceptionBlock, self).__init__()
            self.use_residual = use_residual
            self.return_indices = return_indices
            self.activation = activation
            self.inception_1 = Inception(
                                in_channels=in_channels,
                                n_filters=n_filters,
                                kernel_sizes=kernel_sizes,
                                bottleneck_channels=bottleneck_channels,
                                activation=activation,
                                return_indices=return_indices
                                )
            self.inception_2 = Inception(
                                in_channels=4*n_filters,
                                n_filters=n_filters,
                                kernel_sizes=kernel_sizes,
                                bottleneck_channels=bottleneck_channels,
                                activation=activation,
                                return_indices=return_indices
                                )
            self.inception_3 = Inception(
                                in_channels=4*n_filters,
                                n_filters=n_filters,
                                kernel_sizes=kernel_sizes,
                                bottleneck_channels=bottleneck_channels,
                                activation=activation,
                                return_indices=return_indices
                                )	
            if self.use_residual:
                self.residual = nn.Sequential(
                                    nn.Conv1d(
                                        in_channels=in_channels, 
                                        out_channels=4*n_filters, 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0
                                        ),
                                    nn.BatchNorm1d(
                                        num_features=4*n_filters
                                        )
                                    )

        def forward(self, X):
            if self.return_indices:
                Z, i1 = self.inception_1(X)
                Z, i2 = self.inception_2(Z)
                Z, i3 = self.inception_3(Z)
            else:
                Z = self.inception_1(X)
                Z = self.inception_2(Z)
                Z = self.inception_3(Z)
            if self.use_residual:
                Z = Z + self.residual(X)
                Z = self.activation(Z)
            if self.return_indices:
                return Z,[i1, i2, i3]
            else:
                return Z



    class InceptionTranspose(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU()):
            """
            : param in_channels				Number of input channels (input features)
            : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
            : param kernel_sizes			List of kernel sizes for each convolution.
                                            Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                            This is nessesery because of padding size.
                                            For correction of kernel_sizes use function "correct_sizes". 
            : param bottleneck_channels		Number of output channels in bottleneck. 
                                            Bottleneck wont be used if nuber of in_channels is equal to 1.
            : param activation				Activation function for output tensor (nn.ReLU()). 
            """
            super(InceptionTranspose, self).__init__()
            self.activation = activation
            self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
                                            in_channels=in_channels, 
                                            out_channels=bottleneck_channels, 
                                            kernel_size=kernel_sizes[0], 
                                            stride=1, 
                                            padding=kernel_sizes[0]//2, 
                                            bias=False
                                            )
            self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
                                            in_channels=in_channels, 
                                            out_channels=bottleneck_channels, 
                                            kernel_size=kernel_sizes[1], 
                                            stride=1, 
                                            padding=kernel_sizes[1]//2, 
                                            bias=False
                                            )
            self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
                                            in_channels=in_channels, 
                                            out_channels=bottleneck_channels, 
                                            kernel_size=kernel_sizes[2], 
                                            stride=1, 
                                            padding=kernel_sizes[2]//2, 
                                            bias=False
                                            )
            self.conv_to_maxpool = nn.Conv1d(
                                        in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=1, 
                                        stride=1,
                                        padding=0, 
                                        bias=False
                                        )
            self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
            self.bottleneck = nn.Conv1d(
                                    in_channels=3*bottleneck_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=1, 
                                    stride=1, 
                                    bias=False
                                    )
            self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

            def forward(self, X, indices):
                Z1 = self.conv_to_bottleneck_1(X)
                Z2 = self.conv_to_bottleneck_2(X)
                Z3 = self.conv_to_bottleneck_3(X)
                Z4 = self.conv_to_maxpool(X)

                Z = torch.cat([Z1, Z2, Z3], axis=1)
                MUP = self.max_unpool(Z4, indices)
                BN = self.bottleneck(Z)
                # another possibility insted of sum BN and MUP is adding 2nd bottleneck transposed convolution

                return self.activation(self.batch_norm(BN + MUP))


    class InceptionTransposeBlock(nn.Module):
        def __init__(self, in_channels, out_channels=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU()):
            super(InceptionTransposeBlock, self).__init__()
            self.use_residual = use_residual
            self.activation = activation
            self.inception_1 = InceptionTranspose(
                                in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_sizes=kernel_sizes,
                                bottleneck_channels=bottleneck_channels,
                                activation=activation
                                )
            self.inception_2 = InceptionTranspose(
                                in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_sizes=kernel_sizes,
                                bottleneck_channels=bottleneck_channels,
                                activation=activation
                                )
            self.inception_3 = InceptionTranspose(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_sizes=kernel_sizes,
                                bottleneck_channels=bottleneck_channels,
                                activation=activation
                                )	
            if self.use_residual:
                self.residual = nn.Sequential(
                                    nn.ConvTranspose1d(
                                        in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0
                                        ),
                                    nn.BatchNorm1d(
                                        num_features=out_channels
                                        )
                                    )

        def forward(self, X, indices):
            assert len(indices)==3
            Z = self.inception_1(X, indices[2])
            Z = self.inception_2(Z, indices[1])
            Z = self.inception_3(Z, indices[0])
            if self.use_residual:
                Z = Z + self.residual(X)
                Z = self.activation(Z)
            return Z

    class Flatten(nn.Module):
        def __init__(self, out_features):
            super(Flatten, self).__init__()
            self.output_dim = out_features

        def forward(self, x):
            return x.view(-1, self.output_dim)

    class Reshape(nn.Module):
        def __init__(self, out_shape):
            super(Reshape, self).__init__()
            self.out_shape = out_shape

        def forward(self, x):
            return x.view(-1, *self.out_shape)

    InceptionTime = nn.Sequential(
                        Reshape(out_shape=(7, window_size)),
                        InceptionBlock(
                            in_channels=input_size, 
                            n_filters=32, 
                            kernel_sizes=[5, 11, 23],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        InceptionBlock(
                            in_channels=32*4, 
                            n_filters=32, 
                            kernel_sizes=[5, 11, 23],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        nn.AdaptiveAvgPool1d(output_size=1),
                        Flatten(out_features=32*4*1),
                        nn.Linear(in_features=4*32*1, out_features=num_classes)
            )

    InceptionTime = InceptionTime.to(device)
    InceptionTime

    def train_model_Incept(model, dataloaders, criterion, num_epochs, optimizer):
        since = time.time()

        val_acc_history_Incpt = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):


            if epoch % 20==0 and log_en:
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print('-' * 10)

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_corrects = 0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.long)

                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                        _, preds = torch.max(outputs, 1)

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += labels.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total
                epoch_acc = running_corrects.double() / running_total

                if epoch % 20==0 and log_en:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history_Incpt.append(epoch_acc)

            # print()

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        if log_en:
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)

        # best model 가중치 저장
        # torch.save(best_model_wts, '../output/best_model.pt')
        return model, val_acc_history_Incpt

    # trining 단계에서 사용할 Dataloader dictionary 생성
    dataloaders_dict = {
        'train': train_loader,
        'val': valid_loader
    }

    # loss function 설정
    criterion = nn.CrossEntropyLoss()

    # GRU with attention 모델 학습
    InceptionTime, val_acc_history_Incpt = train_model_Incept(InceptionTime, dataloaders_dict, criterion, num_epochs,
                                                               optimizer=optim.Adam(InceptionTime.parameters(), lr=0.001))

    def test_model_Incept(model, test_loader):
        model.eval()   # 모델을 validation mode로 설정

        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            corrects = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)

                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)

                # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                _, preds = torch.max(outputs, 1)

                # batch별 정답 개수를 축적함
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        # accuracy를 도출함
        test_acc = corrects.double() / total
        if log_en:
            print('[Inception] Testing Acc: {:.4f}'.format(test_acc))

        return test_acc
        
        
        
        
        
    # Incpt Validation Result
    Acc_Incpt_valid = test_model_Incept(InceptionTime, test_loader)
    

    """## 별도의 Test Data로 검증"""
    Acc_Incpt_external_data=0
    if external_test_data:
        x_list = [ext_x]

        x_train_all,y_train_all,x_valid_all,y_valid_all,x_test_all,y_test_all = split_train_test(ext_x, ext_y, x_list, window_size)

        """torch용 data 생성"""
        # no_of_data=400

        x_train, y_train = chunk_merge(x_train_all,y_train_all,window_size)
        x_valid, y_valid = chunk_merge(x_valid_all,y_valid_all,window_size)
        x_test, y_test   = chunk_merge(x_test_all, y_test_all, window_size)

        # train/validation/test 데이터를 window_size 시점 길이로 분할
        datasets = []
        for set in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
            # 전체 시간 길이 설정
            T = set[0].shape[0]

            # 전체 X 데이터를 window_size 크기의 time window로 분할
            # split(array, indices_or_sections) 함수는 자투리 없이 딱 나누어 떨어져야 하므로, 400으로 나눠떨어지도록 자투리 처리, split은 딱 떨어져야 함..
            # array 부분을   set[0].iloc[:window_size * (T // window_size),:] 로 slicing 먼저해주어야 함.
            # windows = np.split(set[0].iloc[:window_size * (T // window_size),:], T // window_size, axis=0)  


            x_sliced = set[0][:window_size * (T // window_size),:]
            x_sliced_transposed = x_sliced.T
            windows = np.split(x_sliced_transposed,T // window_size, axis=1) # axis=1 이면, 가로로 자름.

            # split 하고난, windows는 list형태로 돌아가므로 다시 array 형태로 변환해야 함.
            windows = np.concatenate(windows, axis=0) # axis=0 이면, 세로로 이어붙임.
            if log_en:
                 print("windows_original:",windows.shape)
            # print(windows[:7,:])
            # windows = windows.reshape(window_size,7,-1)
            windows = windows.reshape(-1,7,window_size)

            if log_en:
                 print("windows_reshaped:",windows.shape)
            # print(windows[0,:,:])

            # 전체 y 데이터를 window_size 크기에 맞게 분할
            # labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1) # y는 2차원이므로...
            # labels = np.round(np.mean(np.concatenate(labels, 0), -1))  # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

            labels = np.split(set[1][:window_size * (T // window_size)], T // window_size) # y는 2차원이므로...
            labels = np.round(np.mean((np.concatenate(labels, 0).reshape(-1,window_size)), -1))  # 싯점마다 voting 해서 label 정의한다. 시간축(-1)기준으로 평균 class를 적용하는데, 여기서는 숫자이므로 나중에 round 처리로 함.

            labels = labels.astype(np.long)
            # print(labels[0])

            # shape 확인
            if log_en:
                print(windows.shape,labels.shape)


            # sample data 확인 (마지막 15번 데이터)
            if log_en:
                print("sample data : last label")
                print(windows[-1],labels[-1])

            # 분할된 time window 단위의 X, y 데이터를 tensor 형태로 축적
            datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))



        # train/validation/test DataLoader 구축
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # 모델 검증 (새로운 싯점 데이터 기준 )
        Acc_Incpt_external_data = test_model_Incept(InceptionTime, train_loader)
        
    print("STATUS : Inception 완료")

    return np.max(val_acc_history_Incpt), Acc_Incpt_valid, Acc_Incpt_external_data
