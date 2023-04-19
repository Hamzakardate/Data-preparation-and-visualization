import os
import kivy
import numpy as np
from kivy.app import App
from kivy.uix.label import Label
import pandas as pd
import numpy as np
import keras as kr
import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from keras.models import load_model
import cv2
from kivy.app import App
from kivy.config import Config 
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from skimage.transform import resize
from kivy.uix.filechooser import FileChooserListView, FileChooserIconView
import pathlib

class MyApp(App):

    def importer(self,instance):
        self.camera = cv2.VideoCapture(0)
        return_value, image = self.camera.read()
        cv2.imwrite('opencv_image/opencv.png', image)
        del(self.camera)
    def press(self,instance):
        print("Pressed")
    def pred(self,instance):
        module = load_model('tfmodel.lt')
        b="opencv_image/opencv.png"
        img=plt.imread(b)
        resized_image = resize(img,(32,32,3))
        predictions=module.predict(np.array([resized_image]))
        list=[0,1,2,3,4,5,6,7,8,9]
        class_list=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        X=predictions
        for i in range(10):
          for j in range(10):
            if X[0][list[i]] > X[0][list[j]]:
              temp=list[i]
              list[i]=list[j]
              list[j]=temp
        print(class_list[list[0]])     
        self.title = 'Pred'
        self.box.add_widget(Label(text=f'{class_list[list[0]]}', pos_hint={'center_x':0.3, 'center_y':0.65}))
        return self.box      
    def select_image(self,instance):
        #module = load_model('tfmodel.lt')
        path =self.f.selection[0]
        img=plt.imread(path)
        resized_image = resize(img,(32,32,3))
        tflite_model='model.tflite'
        interpreter = tf.lite.Interpreter(model_path=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        to_predict = np.array(np.array([resized_image]),dtype='float32')
        interpreter.set_tensor(input_details[0]['index'], to_predict)
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])
        a=np.argmax(tflite_results,axis=1)
        #predictions=module.predict(np.array([resized_image]))

        class_list=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        print(class_list[a])     
        self.title = 'Pred'
        self.box.add_widget(Label(text=f'{class_list[list[0]]}', pos_hint={'center_x':0.3, 'center_y':0.5}))
        return self.box
    def build(self):
        
        self.title = 'Free Positioning'
        self.box = FloatLayout(size=(300, 500))
        self.f =FileChooserListView( path = 'C:/Users/ASUS/Desktop/file_app')
        self.box.add_widget(self.f)
        butt1=Button(text='Image Webcam', size_hint=(0.5, 0.2), pos=(0, 0))
        butt1.bind(on_press=self.importer)
        self.box.add_widget(butt1)
        butt2=Button(text='Predect Image', size_hint=(0.5, 0.2), pos=(150, 0))
        butt2.bind(on_press=self.pred)
        self.box.add_widget(butt2)
        butt0=Button(text='Seclect', size_hint=(1, 0.2), pos=(0, 100))
        butt0.bind(on_press=self.select_image)
        self.box.add_widget(butt0)
        return self.box
    
    Config.set('graphics', 'width', '300') 
    Config.set('graphics', 'height', '500')


if __name__ == '__main__':
    MyApp().run()

