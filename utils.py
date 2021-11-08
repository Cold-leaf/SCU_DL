import numpy as np
import cv2

# define the sigmoid function
f = lambda s : 1 / (1 + np.exp(-s))
def sf(x):
    """Compute softmax values for each sets of scores in x."""
    x-=np.max(x)
    exps = np.exp(x)
    return exps / np.sum(exps,axis=0)


# derivative of sigmoid  function
df = lambda s : f(s) * (1-f(s))
#def df(x):
#    """Compute the softmax of vector x in a numerically
#    stable way."""
#    shiftx = x - np.max(x,axis=0)
#    exps = np.exp(shiftx)
#    return exps / np.sum(exps,axis=0)

# Step 4: Define Cost Function
def cost(a, y, beta):
    J = 1/2 * np.sum((a - y)**2)+beta*np.sum(a)
    return J

def cross_entropy_error(a, y, beta):
     #添加一个微小值可以防止负无限大(np.log(0))的发生。
    return -np.sum(y*np.log(a))+beta*np.sum(a)

# Step 5: Define Evaluation Index
def accuracy(a, y):
    #print(a.shape,y.shape)
    mini_batch = a.shape[1]
    idx_a = np.argmax(a, axis=0)
    idx_y = np.argmax(y, axis=0)
    acc = sum(idx_a==idx_y) / mini_batch
    return acc

# Forward computation function
def fc(w, a, b):
     
    z_next = np.dot(w, a)+ b
    #print(z_next.shape)
    a_next = f(z_next)
    #print(a_next.shape)
    #print()
    return a_next, z_next

def fc_sf(w, a, b):
     
    z_next = np.dot(w, a)+ b
    #print(z_next.shape)
    a_next = sf(z_next)
    #print(a_next.shape)
    #print()
    return a_next, z_next

def fc_cap(w, c, a):
    u_mid=w*a 
    s_next=np.sum(c*u_mid,axis=1)
    v_next=f(s_next)
    #z_next = np.dot(w, a)
    #print(z_next.shape)
    #a_next = f(z_next)
    #print(a_next.shape)
    #print()
    return u_mid, s_next, v_next

# Backward computation function
def bc(w, z, delta_next, beta):
    '''
    w.shape [output_dim, input_dim]
    z.shape [feature_dim, batch_size]
    delta_next.shape [feature_dim, batch_size]
    '''
    delta = (np.dot(w.T, delta_next)+beta) * df(z)
    return delta

def bc_cap(w, c, s, delta_next, beta):
    '''
    w.shape [output_dim, input_dim]
    z.shape [feature_dim, batch_size]
    delta_next.shape [feature_dim, batch_size]
    '''
    delta = (np.dot((w*c).T, delta_next)+beta) * df(s).reshape((-1,1))
    return delta

def bc_multi(w, z, a, y, delta_next, beta):
    #print(a.shape,y.shape)
    delta = (np.dot(w.T, delta_next)+ (a - y) + beta) * df(z)
    return delta



def Reverse(trainData):
    bwData=trainData.copy()
    bwData=np.array(bwData,dtype='uint8')

    newData=np.zeros((bwData.shape[0],bwData.shape[1],bwData.shape[2]))
    #lap_5 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    #lap_9 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    for i in range(bwData.shape[0]):
        print("\r{:>6d}/{:<6d}".format(i,bwData.shape[0]),end='')
        bwData[i,:,:]=cv2.GaussianBlur(bwData[i,:,:],(5,5),0)
        bwData[i,:,:]=cv2.Laplacian(bwData[i,:,:],cv2.CV_16S,ksize = 3)#+bwData[i,:,:]
        
        #average=np.mean(bwData[i,:,:])
        #white=np.sum(bwData[i,:,:]>=average)
        #if white>0.53*784:             #阈值
        #    bwData[i,:,:]=2*average-bwData[i,:,:]
            
    bwData=np.where(bwData<1000,bwData,0) 
    bwData=np.where(bwData<255,bwData,255)
    return bwData

def ReverseTile(data):
    redata=data.copy()
    redata=1-redata
    newdata=np.c_[data,redata]

    return newdata

def datablock(data,label,n):
    block={}
    #n=2
    h=int(data.shape[1]/n)
    w=int(data.shape[2]/n)
    for i in range(n):
        for j in range(n):
            block[i*n+j]=data[:,i*h:(i+1)*h,j*w:(j+1)*w]
    newdata=block[0].copy()
    for i in range(1,n*n):
        newdata=np.concatenate((newdata,block[i]),axis=2)

    #Label=np.tile(label,(1,n*n))
    #newlabel=Label.copy()
    #list=np.arange(label.shape[1])
    #for i in range(n*n):
    #    newlabel[:,n*n*list+i]=Label[:,list]
   
    return newdata,block#,newlabel

if __name__ == '__main__':
    data=np.array([[1,2],
                   [2,4]])
    newdata=ReverseTile(data)
    print(data)
    print(newdata)