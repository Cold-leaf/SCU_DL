import math, pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import cost, accuracy, fc, bc, Reverse, ReverseTile, datablock
from PIL import Image

if __name__ == '__main__':
  
    
    trainModel = 'train.pkl'
    with open(trainModel, 'rb') as fo:     # 读取pkl文件数据
        trainData, trainLabels = pickle.load(fo, encoding='bytes')
    
    testModel = 'test.pkl'
    with open(testModel, 'rb') as fo:     # 读取pkl文件数据
        testData, testLabels = pickle.load(fo, encoding='bytes')

    trainLabels =trainLabels.T
    testLabels =testLabels.T

    print(trainData.shape,trainLabels.shape)

    trainData=(trainData[:,:,:,0]*313524 +trainData[:,:,:,1]*615514  +trainData[:,:,:,2]*119538) >> 20
    testData=(testData[:,:,:,0]*313524 +testData[:,:,:,1]*615514  +testData[:,:,:,2]*119538) >> 20

    trainData,trainblock=datablock(trainData, trainLabels,2)      # 分块
    testData,testblock=datablock(testData, testLabels,2)
    print(testData.shape,testLabels.shape)


    train_size = trainData.shape[0]
    X_train = trainData.reshape(train_size,-1).T/255
    trainblock[0]=(trainblock[0].reshape(train_size,-1)).T/255
    trainblock[1]=(trainblock[1].reshape(train_size,-1)).T/255
    trainblock[2]=(trainblock[2].reshape(train_size,-1)).T/255
    trainblock[3]=(trainblock[3].reshape(train_size,-1)).T/255


    test_size = testData.shape[0]
    X_test = testData.reshape(test_size,-1).T/255
    testblock[0]=(testblock[0].reshape(test_size,-1)).T/255
    testblock[1]=(testblock[1].reshape(test_size,-1)).T/255
    testblock[2]=(testblock[2].reshape(test_size,-1)).T/255
    testblock[3]=(testblock[3].reshape(test_size,-1)).T/255

    print(X_train.shape)

    L_list=[7]
    size_list=[128,32]
    
    

    for L in L_list:
        # Step 2: Network Architecture Design
        # define number of layers       
        layer_size=[]
        layer_size.append(14*14)
        #layer_size.append(28*28)
        for i in range(L-5):
            layer_size.append(size_list[i])
        layer_size.append(10)
        print("\nL="+str(L),layer_size)
        
        
        
        # Step 3: Initializing Network Parameters
        # initialize weights
        w = {}
        b = {}

        for i in range(1,4):
            w[i] = 0.5*np.random.randn(196, 392)
            b[i] = 0
        for i in range(4, L):
            w[i] = 0.5*np.random.randn(layer_size[i-3], layer_size[i-1-3])            
            b[i] = 0

        lr = 0.05 # initialize learning rate
        beta=1e-4
        drop=0.2
        # Step 4,5 see utils.py

        # Step 6: Train the Network
        J = []   
        acc = []
        acc_t = []

        max_epoch = 200 # number of training epoch 200
        mini_batch = 100 # number of sample of each mini batch 100
        for epoch_num in range(max_epoch):

            Acc = [] # array to store accuracy of each mini batch
            idxs = np.random.permutation(train_size)
            j = []
            for k in range(math.ceil(train_size/mini_batch)):

                start_idx = k*mini_batch 
                end_idx = min((k+1)*mini_batch, train_size) 

                a, z, delta = {}, {}, {}
                A={}    # 和外部输入合并后的输入
                batch_indices = idxs[start_idx:end_idx]               
                a[1] = trainblock[0][:, batch_indices]                
                y = trainLabels[:, batch_indices]
     
                # forward computation
                for i in range(1,4):                    
                    A[i]=np.r_[trainblock[i][:, batch_indices],a[i]]
                    a[i+1], z[i+1] = fc(w[i], A[i], b[i])
                for i in range(4 , L):                    
                    a[i+1], z[i+1] = fc(w[i], a[i], b[i])
                
                delta[L] = (a[L] - y + beta) * (a[L]*(1-a[L])) 

                # backward computation
                for i in range(L-1, 3, -1):                   
                    delta[i] = bc(w[i], z[i], delta[i+1], beta)
            
                for i in range(3, 1, -1):
                    delta[i] = bc(w[i][:,196:], z[i], delta[i+1], beta)

                # update weights
                for i in range(1, 4):
                    grad_w = np.dot(delta[i+1], A[i].T)
                    w[i] = w[i] - lr*grad_w
                for i in range(4, L):
                    grad_w = np.dot(delta[i+1], a[i].T)
                    w[i] = w[i] - lr*grad_w
                    
                
                Acc.append(accuracy(a[L], y))
                j.append(cost(a[L], y, beta))
            
            acc.append(sum(Acc)/len(Acc))
            J.append(sum(j)/train_size)

            a_test,z_test={},{}
            A_test={}
            a_test[1] = testblock[0]           
            y_test = testLabels


            for i in range(1,4):                    
                A_test[i]=np.r_[testblock[i],a_test[i]]
                a_test[i+1], z_test[i+1] = fc(w[i], A_test[i], b[i])
            for i in range(4 , L):                    
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