from buildmodel import Build_Model
from mergesort import MergeSort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import os

if __name__ == '__main__': 
    model_name = "densenet" #change to inception for inception v3 training
    num_classes = 3
    batch_size = 8
    num_epochs = 15
    feature_extract = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    IMG_WIDTH=224
    IMG_HEIGHT=224
    img_folder='test'

    build_model = Build_Model(device)
    model_ft, input_size = build_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
    
    print(model_ft)
    
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load('densenet_0.pth'))
   
    model_ft.eval()
    
    inception_res = []
    list_im = []
    list_dir = []

    def create_dataset(img_folder):
    
        img_data_array=[]
        class_name=[]
        im_dir_name = []
    
        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):
        
                image_path= os.path.join(img_folder, dir1,  file)
                #print(image_path)
                im_dir_name.append(image_path)
                image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                image=np.array(image)
                image = image.astype('float32')
                image /= 255 
                img_data_array.append(image)
                class_name.append(dir1)
        return img_data_array, class_name, im_dir_name
    
    img_data, class_name, im_dir_name =create_dataset('./test')
    
    target_dict={k: v for v, k in enumerate(np.unique(class_name))}
    target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]

    def im_normalize(x):
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
        
        x = torch.from_numpy(np.array(x))
        x = x.type(torch.float32)
        x = x.permute(-1, 0, 1)
        x = (x - MEAN[:, None, None]) / STD[:, None, None]
        return x
  
    
            
    for idx, img in enumerate(img_data):
        img_dict = im_normalize(img)
        list_dir.append(im_dir_name[idx])
        img = img_dict[np.newaxis,:,:,:]
        image_tensor = Variable(img).to(device, dtype=torch.float)
        sr_image = model_ft(image_tensor).cpu()
        probabilities = torch.nn.functional.softmax(sr_image[0], dim=0)
        tmp = probabilities.detach().numpy()
        print(tmp)
        inception_res.append(list(tmp).index(max(tmp)))
        list_im.append(img_dict.detach().numpy())
    
    #Currently showing for densenet
    #use weights of inception v3 and change the string to Inception v3 evaluation metrics:  
    print("Densenet evaluation metrics:")
    print("Accuracy:", accuracy_score(target_val, inception_res))
    print("F1 Score:", f1_score(target_val, inception_res, average='macro'))
    print("Precision:", precision_score(target_val, inception_res, average='macro'))
    print("Recall:", recall_score(target_val, inception_res, average='macro'))


    arr = inception_res
    list_im_arr = list_dir
    n = len(arr)
    print("Given array is")
    for i in range(n):
        print("%d" % arr[i],end=" ")
    print()
    for i in list_im_arr:
        print(i)
    merger = MergeSort()       
    merger.mergeSort(list_im_arr, arr, 0, n-1)
    print("\n\nSorted array is")
    for i in range(n):
        print("%d" % arr[i],end=" ")
    print()
    for i in list_im_arr:
        print(i)