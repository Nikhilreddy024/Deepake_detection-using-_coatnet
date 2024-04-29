### Deepfake_detection-using-Coatnet

### Weights
big_model(coatnet-2).pth - 631MB. <br />
small_model(coatnet-0).pth - 200MB. <br />


### Preprocessing

---python FaceExtraction.py 

&nbsp;&nbsp;&nbsp;Face extraction from video. <br /> 
&nbsp;&nbsp;&nbsp;The code works for DFDC dataset. You can test it using the sample data provided. 

Move the DFDC data chunks to the 'preprocessing/data' folder, then execute FaceExtraction.py to extract faces from videos, storing them in the "/face_data" directory.


### Train Coatnet


---python coatnet_train.py -e 10 -s 'g' -l 0.0001 -w 0.0000001 -d "path_to_folder" 

To train the model on your own you can use the following parameters:<br />
&nbsp;&nbsp;e: epoch <br/>
&nbsp;&nbsp;s: session - **(g)** - GPU or **(t)** - TPU <br/>
&nbsp;&nbsp;w: weight decay  default= 0.0000001 <br/>
&nbsp;&nbsp;l: learning rate default=0.001 <br/>
&nbsp;&nbsp;d: path file <br/>
&nbsp;&nbsp;b: batch size, defualt=32 <br/>

The model is fed the data from the '/face_data' directory and stores its weights in the '/weights' folder.


### Predict Coatnet


---python coatnet_prediction.py

&nbsp;&nbsp;&nbsp; Predicts whether a video is Deepfake or not.<br />
&nbsp;&nbsp;&nbsp; Prediction value <0.5 - REAL <br />
&nbsp;&nbsp;&nbsp; Prediction value >=5  - FAKE


---python distribution.py

If you intend to use only single chunk of the dfdc dataset.use this file to distribute single dfdc folder into multiple dfdc folders maintaining integrity of real and fake videos.


'''Check out our blog, "https://medium.com/@jithendra.katta/deepfake-detection-using-convolutions-with-attention-coatnet-78c8b9f4d69d" to learn more about the code.'''

##Credits:
This Work is truly inspired by the work done by CViT: https://github.com/erprogs/CViT <br/>
and an experiment of implemeting CoAtNet: https://arxiv.org/abs/2106.04803





