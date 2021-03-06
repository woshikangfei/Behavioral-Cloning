import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def preprocess(image):
    image = image[30:-25,:,:]
    image = cv2.resize(image,(200,66),cv2.INTER_AREA)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
    return image

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    images = np.asarray(image)

    x_train_processed_image = images[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    x_train_processed_image = np.array([preprocess(image) for image in x_train_processed_image])
    #transformed_image_array = cv2.resize((image_array[30:135, :, :]),(200,66))

    #add
    #image = img.convert('RGB')
    #img = img.resize((img_height,img_width ))
    #x_train_processed_image=np.zeros((len(img),img_width,img_height,3))
    #x_train_processed_image[0] =x_train_processed_image
    #image_array=x_train_processed_image

    #transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    #images = images[40:,:,:]
    #images = cv2.resize(images,(200,66),interpolation=cv2.INTER_CUBIC)
    #images =cv2.cvtColor(images,cv2.COLOR_BGR2YUV)
    #images= images.astype('float32')
    #x_train_processed_image=np.zeros((1,66,200,3))
    #x_train_processed_image[0]=images
    #print(1)
    
    
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(x_train_processed_image, batch_size=1))
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    print(weights_file)
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
