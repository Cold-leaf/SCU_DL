import math, pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import cost, accuracy, fc, bc, Reverse, ReverseTile, datablock
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

    print(trainData.shape,trainLabels.shape)

    trainData=(trainData[:,:,:,0]*313524 +trainData[:,:,:,1]*615514  +trainData[:,:,:,2]*119538) >> 20
    testData=(testData[:,:,:,0]*313524 +testData[:,:,:,1]*615514  +testData[:,:,:,2]*119538) >> 20


    train_size = trainData.shape[0]
    X_train = trainData.reshape(train_size,-1).T/255
  

    test_size = testData.shape[0]
    X_test = testData.reshape(test_size,-1).T/255


    print(X_train.shape)

 
    
    Size_list1=[784,128,32]
    Size_list2=[0,0,128,32,10]
    L=5

   
    
        
    # Step 3: Initializing Network Parameters
    # initialize weights
    w1 = {}
    w2 = {}
    w3,w4={},{}
    W = {}
    b = {}

    for i in range(1, 3):
        w1[i]= 0.5*np.random.randn(Size_list1[i], Size_list1[i-1])
        w2[i]= 0.1*np.random.randn(Size_list1[i], Size_list1[i-1])
        w3[i]= 0.5*np.random.randn(Size_list1[i], Size_list1[i-1])
        w4[i]= 0.1*np.random.randn(Size_list1[i], Size_list1[i-1])
    for i in range(3, L):
        W[i]= 0.5*np.random.randn(Size_list2[i], Size_list2[i-1])
    for i in range(1, L):
        b[i] = 0

    lr = 0.05 # initialize learning rate
    beta=1e-4
    drop=0.2
    # Step 4,5 see utils.py

    # Step 6: Train the Network
    J = [] # array to store cost of each mini batch   
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
            a1,z1,a2,z2={},{},{},{}
            a3,z3,a4,z4={},{},{},{}
            delta1,delta2={},{}
            delta3,delta4={},{}

            batch_indices = idxs[start_idx:end_idx]
            a1[1] = X_train[:, batch_indices]
            a2[1] = X_train[:, batch_indices]
            a3[1] = X_train[:, batch_indices]
            a4[1] = X_train[:, batch_indices]
            y = trainLabels[:, batch_indices]
                
                
            # 子网络
            for i in range(1, 3):
               a1[i+1], z1[i+1] = fc(w1[i], a1[i], b[i])
               a2[i+1], z2[i+1] = fc(w2[i], a2[i], b[i])
               a3[i+1], z3[i+1] = fc(w3[i], a3[i], b[i])
               a4[i+1], z4[i+1] = fc(w4[i], a4[i], b[i])

            # 合并输入
            a[3]=np.r_[a1[3],a2[3],a3[3],a4[3]]
            z[3]=np.r_[z1[3],z2[3],z3[3],z4[3]]

            for i in range(3,L):
                a[i+1], z[i+1] = fc(W[i], a[i], b[i])

                
            delta[L] = (a[L] - y + beta) * (a[L]*(1-a[L])) 
            # backward computation
            for i in range(L-1, 2, -1):
                delta[i] = bc(W[i], z[i], delta[i+1], beta)
            
            delta1[2]=bc(w1[2],z1[2],delta[3][0*Size_list1[2]:1*Size_list1[2],:],beta)
            delta2[2]=bc(w2[2],z2[2],delta[3][1*Size_list1[2]:2*Size_list1[2],:],beta)
            delta3[2]=bc(w3[2],z3[2],delta[3][2*Size_list1[2]:3*Size_list1[2],:],beta)
            delta4[2]=bc(w4[2],z4[2],delta[3][3*Size_list1[2]:4*Size_list1[2],:],beta)
                
            # update weights
            w1[1]=w1[1]-lr*np.dot(delta1[2],a1[1].T)
            w1[2]=w1[2]-lr*np.dot(delta[3][0*Size_list1[2]:1*Size_list1[2],:],a1[2].T)

            w2[1]=w2[1]-lr*np.dot(delta2[2],a2[1].T)
            w2[2]=w2[2]-lr*np.dot(delta[3][1*Size_list1[2]:2*Size_list1[2],:],a2[2].T)

            w3[1]=w3[1]-lr*np.dot(delta3[2],a3[1].T)
            w3[2]=w3[2]-lr*np.dot(delta[3][2*Size_list1[2]:3*Size_list1[2],:],a3[2].T)

            w4[1]=w4[1]-lr*np.dot(delta4[2],a4[1].T)
            w4[2]=w4[2]-lr*np.dot(delta[3][3*Size_list1[2]:4*Size_list1[2],:],a4[2].T)
            for i in range(3, L):
                grad_w = np.dot(delta[i+1], a[i].T)
                W[i] = W[i] - lr*grad_w
                
            Acc.append(accuracy(a[L], y))
            j.append(cost(a[L], y, beta))
        #lr=lr*0.995
        acc.append(sum(Acc)/len(Acc))
        J.append(sum(j)/train_size)

        a_test,z_test={},{}
        a1_test,z1_test={},{}
        a2_test,z2_test={},{}
        a3_test,z3_test={},{}
        a4_test,z4_test={},{}
        a1_test[1] = X_test   
        a2_test[1] = X_test  
        a3_test[1] = X_test  
        a4_test[1] = X_test
        y_test = testLabels

        for i in range(1, 3):
            a1_test[i+1], z1_test[i+1] = fc(w1[i], a1_test[i], b[i])
            a2_test[i+1], z2_test[i+1] = fc(w2[i], a2_test[i], b[i])
            a3_test[i+1], z3_test[i+1] = fc(w3[i], a3_test[i], b[i])
            a4_test[i+1], z4_test[i+1] = fc(w4[i], a4_test[i], b[i])
        a_test[3]=np.r_[a1_test[3],a2_test[3],a3_test[3],a4_test[3]]
        for i in range(3,L):
            a_test[i+1], z_test[i+1] = fc(W[i], a_test[i], b[i])
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
