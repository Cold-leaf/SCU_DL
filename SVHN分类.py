import math, pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import cost, accuracy, fc, fc_sf, bc, cross_entropy_error, Reverse, ReverseTile, datablock
from PIL import Image

if __name__ == '__main__':
    ## Step 1: Data Preparation
    
    trainModel = 'train.pkl'
    with open(trainModel, 'rb') as fo:     # 读取pkl文件数据
        trainData, trainLabels = pickle.load(fo, encoding='bytes')
    
    testModel = 'test.pkl'
    with open(testModel, 'rb') as fo:     # 读取pkl文件数据
        testData, testLabels = pickle.load(fo, encoding='bytes')

    trainLabels =trainLabels.T
    testLabels =testLabels.T

    trainData=(trainData[:,:,:,0]*313524 +trainData[:,:,:,1]*615514  +trainData[:,:,:,2]*119538) >> 20      # 灰度化
    testData=(testData[:,:,:,0]*313524 +testData[:,:,:,1]*615514  +testData[:,:,:,2]*119538) >> 20

    #trainData,trainblock=datablock(trainData, trainLabels,2)      # 分块序列输入
    #testData,testblock=datablock(testData, testLabels,2)
    #print(testData.shape,testLabels.shape)

    #trainData=Reverse(trainData)       # 滤波
    #testData=Reverse(testData)

    train_size = trainData.shape[0]  
    X_train = trainData.reshape(train_size,-1).T/255

    #trainblock[1]=(trainblock[1].reshape(train_size,-1)).T     # 图像分块
    #trainblock[2]=(trainblock[2].reshape(train_size,-1)).T
    #trainblock[3]=(trainblock[3].reshape(train_size,-1)).T
    #trainblock[4]=(trainblock[4].reshape(train_size,-1)).T

    #X_train=ReverseTile(X_train)              # 反转扩充
    #trainLabels=np.c_[trainLabels,trainLabels]
    #train_size=X_train.shape[1]

    test_size = testData.shape[0]
    X_test = testData.reshape(test_size,-1).T/255


    L_list=[5]
    size_list=[512,128,64]
    
    

    for L in L_list:
        # Step 2: Network Architecture Design
        # define number of layers       
        layer_size=[]
        layer_size.append(28*28)
        for i in range(L-2):
            layer_size.append(size_list[i])
        layer_size.append(10)
        print("\nL="+str(L),layer_size)

        
        # Step 3: Initializing Network Parameters
        # initialize weights
        w = {}
        vw = {}       # 动量
        vb = {}
        b = {}       # 偏置
        for i in range(1, L):
            #np.random.seed(i)
            w[i] = 0.5*np.random.randn(layer_size[i], layer_size[i-1])
            #vw[i] = np.zeros(w[i].shape)
            #vb[i] = 0
            b[i] = 0

        lr = 0.05   # initialize learning rate
        dr=0.5      # 动量系数
        beta=1e-4
        #drop=0.3   # drop比例
        # Step 4,5 see utils.py

        # Step 6: Train the Network
        J = []      # 一次迭代的平均loss
        acc = []    # 训练集acc
        acc_t = []  # 测试集acc

        max_epoch = 200 # number of training epoch 200
        mini_batch = 100 # number of sample of each mini batch 100
        for epoch_num in range(max_epoch):
            Acc = [] # array to store accuracy of each mini batch
            idxs = np.random.permutation(train_size)
            j = []
            for k in range(math.ceil(train_size/mini_batch)):
                
                #a_c={}         # 引入dropout时copy原神经元
                #w_c={}
                #row={}
                start_idx = k*mini_batch 
                end_idx = min((k+1)*mini_batch, train_size) 

                a, z, delta = {}, {}, {}
                batch_indices = idxs[start_idx:end_idx]
                a[1] = X_train[:, batch_indices]
                
                y = trainLabels[:, batch_indices]
                
                
                #a_c[1]=a[1].copy()
                #w_c[1]=w[1].copy()
                #row[1]=np.random.randint(a_c[1].shape[0],size=(int(drop*a_c[1].shape[0])))    # 随机选择drop的神经元
                #a_c[1][row[1],:]=0                
                #a[2], z[2] = fc(w[1], a_c[1], b[1])
                # forward computation
                for i in range(1, L):
                    a[i+1], z[i+1] = fc(w[i], a[i], b[i])
                #a[L], z[L] = fc_sf(w[L-1], a[L-1], b[L-1])         # 最后一层使用softmax作为激活函数时

                

                #delta[L] = (a[L] - y + beta)           # 最后一层使用softmax作为激活函数时
                delta[L] = (a[L] - y + beta) * (a[L]*(1-a[L])) 
                # backward computation
                for i in range(L-1, 1, -1):                   
                    delta[i] = bc(w[i], z[i], delta[i+1], beta)
                    
                # update weights
                for i in range(1, L):
                    
                    grad_w = np.dot(delta[i+1], a[i].T)
                    #vw[i]=dr*vw[i]+(1-dr)*grad_w      # Momentum
                    #w[i]=w[i]-lr*vw[i]
                    w[i] = w[i] - lr*grad_w
                    
                    grad_b = np.mean(delta[i+1],axis=1).reshape(-1,1)
                    #vb[i]=dr*vb[i]+(1-dr)*grad_b
                    #b[i] = b[i] - lr*vb[i]
                    b[i] = b[i] - lr*grad_b
                #w[1][:,row[1]]=w_c[1][:,row[1]]    # 还原drop神经元对应的w
                

                Acc.append(accuracy(a[L], y))
                j.append(cost(a[L], y, beta))
                #j.append(cross_entropy_error(a[L], y, beta))
            #lr=lr*0.995
            acc.append(sum(Acc)/len(Acc))
            J.append(sum(j)/train_size)

            a_test,z_test={},{}
            a_test[1] = X_test           
            y_test = testLabels

            #a_c[1]=a_test[1].copy()
            #w_c[1]=w[1].copy()
            #row[1]=np.random.randint(a_c[1].shape[0],size=(int(drop*a_c[1].shape[0])))    
            #a_c[1][row[1],:]=0                
            #a_test[2], z_test[2] = fc(w[1], a_c[1], b[1])
            # forward computation 
            for i in range(1, L):
                a_test[i+1], z_test[i+1] = fc(w[i], a_test[i], b[i])
            acc_t.append(accuracy(a_test[L], y_test))
            print("epoch:{:3d}".format(epoch_num), \
                " training acc:{:>6.0f}/{:6d}({:<5.4f})".format(acc[epoch_num]*train_size,train_size,acc[epoch_num]),\
                " training loss:{:<5.4f}".format(J[epoch_num]),\
                ' test acc:{:<5.4f}'.format(acc_t[epoch_num]),\
                ' lr={:.6f}'.format(lr))

        #绘图
        plt.subplot(1,2,1)
        plt.title("Accuracy",size=15)
        plt.grid(axis='y',ls='-.')
        plt.xlabel("epoch")
        plt.ylabel("Acc")
        plt.ylim(0.7,1)
        plt.plot(np.arange(max_epoch),acc,label="TrainAcc",c="r")
        plt.plot(np.arange(max_epoch),acc_t,label="TestAcc")                    
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.title("Cost",size=15)
        plt.grid(axis='y',ls='-.')
        plt.xlabel("epoch")
        plt.ylabel("Cost")
        plt.plot(np.arange(max_epoch),J,label="TrainLoss",c="r")
        plt.legend()

        plt.show()
   