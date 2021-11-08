import os
import h5py
from PIL import Image
import cv2
import glob
import pickle
import numpy as np
 
def get_attrs(digit_struct_mat_file, index):
    """
    Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
    """
    attrs = {}
    f = digit_struct_mat_file
    item = f['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = f[item][key]
        values = [f[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs

for pathname in ['train','test']:


    Data=[]
    Label=[]
 
    path_to_dir = pathname
    path_to_digit_struct_mat_file = os.path.join(path_to_dir, 'digitStruct.mat')
 
    path_to_dir = pathname   #存放图片的文件夹路径
    paths = glob.glob(os.path.join(path_to_dir, '*.png'))
    #paths.sort()
    #print(len(paths)) 
    num = 0
    for i in paths:
    
        path_to_image_file = os.path.join(i)
        index = int(path_to_image_file.split('\\')[-1].split('.')[0]) - 1   # train中的序号（非label）
    
        #print(index, path_to_image_file)
        img = cv2.imread(path_to_image_file)

    
        with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
            attrs = get_attrs(digit_struct_mat_file, index)

            length = len(attrs['label'])
        
            attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x],
                                                               [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
       
        for attr_left, attr_top, attr_width, attr_height,label in zip(attrs_left, attrs_top, attrs_width, attrs_height,attrs['label']):
   
            cut = img[attr_top:attr_top+attr_height,attr_left:attr_left+attr_width]
            #print(cut.shape)
        
            #cutimg = Image.fromarray(cut)
    #         display(image_show_11)
            if (img is None) or (cut.shape[1] == 0) or (cut.shape[0]== 0):
                #print("--->",i)
                attr_left=0
                cut = img[attr_top:attr_top+attr_height,attr_left:attr_left+attr_width]
                #print(i,attr_left, attr_top, attr_width, attr_height, label)
            #else:
            reimg=cv2.resize(cut,(28,28))
            Data.append(reimg)
            if int(label) == 10 :
                #cv2.imwrite("result/"+pathname+"/0/"+str(num)+"_"+str(int(label-10))+".jpg",reimg)
                Label.append([1,0,0,0,0,0,0,0,0,0])
            if int(label) == 1 :
                #cv2.imwrite("result/"+pathname+"/1/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,1,0,0,0,0,0,0,0,0])
            if int(label) == 2 :
                #cv2.imwrite("result/"+pathname+"/2/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,0,1,0,0,0,0,0,0,0])
            if int(label) == 3 :
                #cv2.imwrite("result/"+pathname+"/3/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,0,0,1,0,0,0,0,0,0])
            if int(label) == 4 :
                #cv2.imwrite("result/"+pathname+"/4/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,0,0,0,1,0,0,0,0,0])
            if int(label) == 5 :
                #cv2.imwrite("result/"+pathname+"/5/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,0,0,0,0,1,0,0,0,0])
            if int(label) == 6 :
                #cv2.imwrite("result/"+pathname+"/6/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,0,0,0,0,0,1,0,0,0])
            if int(label) == 7 :
                #cv2.imwrite("result/"+pathname+"/7/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,0,0,0,0,0,0,1,0,0])
            if int(label) == 8 :
                #cv2.imwrite("result/"+pathname+"/8/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,0,0,0,0,0,0,0,1,0])
            if int(label) == 9 :
                #cv2.imwrite("result/"+pathname+"/9/"+str(num)+"_"+str(int(label))+".jpg",reimg)
                Label.append([0,0,0,0,0,0,0,0,0,1])
            
        num = num + 1
        print("\r正在裁剪第{:>6d}/{:<6d}张图片".format(num,len(paths)),end='')

    model_name = pathname+'.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump([np.array(Data), np.array(Label)], f)
    print("\nmodel saved to {}".format(model_name))