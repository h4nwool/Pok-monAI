# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:44:43 2020

@author: daum
"""

from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/input")
def input():
    return render_template("input.html")
    
@app.route("/test")
def test():
    return render_template("input_han.html")

@app.route("/output", methods = ["POST","GET"])
def output():
    if  request.method =="POST":
        inputimg =  request.files['user_img']
	
        import cnn_model
        import keras
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        from keras.models import load_model
        import cv2    
        from keras import backend as K
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.models import load_model

        im_rows = 64 # 이미지의 높이
        im_cols = 64 # 이미지의 너비
        im_color = 3 # 이미지의 색공간
        in_shape = (im_rows, im_cols, im_color)
        nb_classes = 7

        LABELS = ["꼬부기", "치코리타", "잠만보", "뽀뽀라", "케이시", "푸린","리아코"]
        # 저장한 CNN 모델 읽어 들이기
        model = cnn_model.get_model(in_shape, nb_classes)

        # CNN 모델 읽기
        model = load_model('photos-cnn-model.h5')
  
        # 학습된 데이타 불러오기
        model.load_weights('photos-cnn-weight.hdf5')


        # MLP로 학습한 이미지 데이터에 형태 맞추기
        # im = im.reshape(in_shape).astype('float32') / 255


            # 이미지 읽어 들이기
        img = Image.open(inputimg)#<class 'PIL.JpegImagePlugin.JpegImageFile'>

        #img = img.convert("RGB") # 색공간 변환하기
        

        #img = img.resize((im_cols, im_rows)) # 크기 변경하기
            
            # 데이터 변환하기
        n_img = np.asarray(img)
        n_img = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("test_before.jpg",n_img)  
        
        
                    
        cascade_file = "haarcascade_frontalface_alt.xml" # 정면 얼굴 검출
        cascade = cv2.CascadeClassifier(cascade_file)
        img_gray = cv2.cvtColor(n_img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(30,30)) # miniSize - 얼굴 인식 영역의 최소 크기 지정
        if len(face_list) == 1:
            for (x,y,w,h) in face_list:
                #print("얼굴의 좌표 =", x, y, w, h)
                dst = n_img.copy()####이미지복사해서
                dst = n_img[y:y+h,x:x+w] ###자르기
                cv2.imwrite("test_after.jpg",dst)
                dst = Image.fromarray(dst, 'RGB')
                
            dst = dst.resize((im_cols, im_rows))
            dst = np.asarray(dst)
            cv2.imwrite("test_resize.jpg",dst)
            dst = dst.reshape(-1, im_rows, im_cols, im_color)
            dst = dst / 255
           

                # 예측하기
            pre = model.predict([dst])[0]
            K.clear_session()  
            idx = pre.argmax()
           

            result_pm = LABELS[idx]
  
            
        elif len(face_list) != 1:
            result_pm = "메타몽"

            
 
        return render_template("output.html", result_pm = result_pm )
            


    
if __name__ == "__main__":
    app.run(debug= True)
    
