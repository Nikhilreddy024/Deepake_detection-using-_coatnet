import sys
import cv2
import torch
import numpy as np
from time import perf_counter
from torchvision import transforms
import face_recognition
import random

sys.path.insert(1,'model')

from coatnet import CoAtNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#dfdc dataset's mean and standard deviation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


#Normalisation
normalize_transform = transforms.Compose([
        transforms.Normalize(mean, std)]
)

ran = random.randint(0,400)
ran_min = abs(ran-1)


#Coatnet-0
#model = CoAtNet(image_size=(224, 224), in_channels=3, num_blocks=[2, 2, 3, 5, 2], channels=[64, 96, 192, 384, 768], num_classes=2)
#-----------------
#Coatnet-2
model = CoAtNet(image_size=(224, 224), in_channels=3, num_blocks=[2, 2, 6, 14, 2], channels = [128, 128, 256, 512, 1026], num_classes=2)

model.to(device)

checkpoint = torch.load('weight/big_model(coatnet-2).pth') # for GPU
model.load_state_dict(checkpoint['state_dict'])
_ = model.eval()


def predict_on_video():
    
    #Give the path of the video file here
    dec = predict(r"sample_data_for_prediction/pfake.mp4")
    return dec

    

# face_recognition for face extraction
# It performs better than blaze_face

def face_face_rec(frame, face_tensor_face_rec):
    
    face_locations = face_recognition.face_locations(frame)
    temp_face = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    count=0
    for face_location in face_locations:
        if count<5:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            temp_face[count]=face_image
            count+=1
    if count == 0:
        return [],0
    return temp_face[:count], count


y_pred=0

def predict(filename):
    
    face_tensor_face_rec = np.zeros((30, 224, 224, 3), dtype=np.uint8)
    
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame_number = 0
    frame_count=int(length*0.1)
    frame_jump = 5 
    start_frame_number = 0

    loop = 0
    count_face_rec = 0
    
    while cap.isOpened() and loop<frame_count:
        loop+=1
        success, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        
        if success:
            face_rec,count = face_face_rec(frame, face_tensor_face_rec)
            
            if len(face_rec) and count>0:
                kontrol = count_face_rec+count
                for f in face_rec:
                    if count_face_rec<=kontrol and (count_face_rec<29):
                        face_tensor_face_rec[count_face_rec] = f
                        count_face_rec+=1

            start_frame_number+=frame_jump
    

    store_rec= face_tensor_face_rec[:count_face_rec]
    
    dfdc_tensor=store_rec
    
    dfdc_tensor = torch.tensor(dfdc_tensor, device=device).float()

    # Preprocess the images.
    dfdc_tensor = dfdc_tensor.permute((0, 3, 1, 2))

    for i in range(len(dfdc_tensor)):
        dfdc_tensor[i] = normalize_transform(dfdc_tensor[i] / 255.)
    
    if not len(non_empty(dfdc_tensor, df_len=-1, lower_bound=-1, upper_bound=-1, flag=False)):
        return torch.tensor(0.5).item()
        
    dfdc_tensor = dfdc_tensor.contiguous()
    df_len = len(dfdc_tensor)
    
    with torch.no_grad(): 
        
        thrtw =32
        if df_len<33:
            thrtw =df_len  
        y_predCN = model(dfdc_tensor[0:thrtw])
        
        if df_len>32:
            dft = non_empty(dfdc_tensor, df_len, lower_bound=32, upper_bound=64, flag=True)
            if len(dft):
                y_predCN = pred_tensor(y_predCN, model(dft))
        if df_len>64:
            dft = non_empty(dfdc_tensor, df_len, lower_bound=64, upper_bound=90, flag=True)
            if len(dft):
                y_predCN = pred_tensor(y_predCN, model(dft))
        
        dec = pre_process_prediction(pred_sig(y_predCN))
        print('CN', filename, "Prediction:",dec.item())
        print("in predict")
        return dec.item()

def non_empty(dfdc_tensor, df_len, lower_bound, upper_bound, flag):
    
    thrtw=df_len
    if df_len>=upper_bound:
        thrtw=upper_bound
        
    if flag==True:
        return dfdc_tensor[lower_bound:thrtw]
    elif flag==False:
        return dfdc_tensor
        
    return []
    
def pred_sig(dfdc_tensor):
    return torch.sigmoid(dfdc_tensor.squeeze())

def pred_tensor(dfdc_tensor, pre_tensor):
    return torch.cat((dfdc_tensor,pre_tensor),0)

def pre_process_prediction(y_pred):
    f=[]
    r=[]
    if len(y_pred)>2:
        for i, j in y_pred:
            f.append(i)
            r.append(j)
        f_c = sum(f)/len(f)
        r_c= sum(r)/len(r)
        if f_c>r_c:
            return f_c
        else:
            r_c = abs(1-r_c)
            return r_c
    else:
        return torch.tensor(0.5)
    
start_time = perf_counter()
predictions = predict_on_video()
print(predictions)
if(predictions>0.5):
    print('Fake')
else:
    print('Real')    
end_time = perf_counter()
print("--- %s seconds ---" % (end_time - start_time))




    
