from jetbotmini import Camera
from jetbotmini import bgr8_to_jpeg
camera = Camera.instance(width=300, height=300)

import numpy as np
global color_lower
color_lower=np.array([156,43,46])
global color_upperv
color_upper = np.array([180, 255, 255])

import torch
import torchvision
import torch.nn.functional as F
import cv2
import traitlets
import ipywidgets.widgets as widgets
import numpy as np

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

from jetbotmini import Robot

robot = Robot()

import PID

global turn_gain
turn_gain = 1.7
global turn_gain_pid
turn_gain_pid = PID.PositionalPID(0.15, 0, 0.05)

color_lower=np.array([156,43,46])
color_upper = np.array([180, 255, 255])

color_lower=np.array([16,43,46])
color_upper = np.array([34, 255, 255])

from  matplotlib import pyplot as plt
%matplotlib inline
from IPython import display

for i in range(10):
    frame = camera.value
    frame = cv2.resize(frame, (300, 300))
    frame_=cv2.GaussianBlur(frame,(5,5),0)                    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,color_lower,color_upper)  
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    mask=cv2.GaussianBlur(mask,(3,3),0)  
    plt.imshow(mask)
    plt.show()
    display.clear_output(wait=True)

from jetbotmini import bgr8_to_jpeg
from IPython.display import display

image_widget = widgets.Image(format='jpeg', width=300, height=300)
speed_widget = widgets.FloatSlider(value=0.4, min=0.0, max=1.0, description='speed')

display(widgets.VBox([
    widgets.HBox([image_widget]),
    speed_widget,
]))

width = int(image_widget.width)
height = int(image_widget.height)
       
def execute(change):
    global turn_gain
    target_value_speed = 0
   
    #Update picture values
    frame = camera.value
    frame = cv2.resize(frame, (300, 300))
    frame_=cv2.GaussianBlur(frame,(5,5),0)                    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,color_lower,color_upper)  
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    mask=cv2.GaussianBlur(mask,(3,3),0)    
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
           
    # Target detected
    if len(cnts)>0:
        cnt = max (cnts,key=cv2.contourArea)
        (color_x,color_y),color_radius=cv2.minEnclosingCircle(cnt)
        if color_radius > 10:
            # Mark the detected color
            cv2.circle(frame,(int(color_x),int(color_y)),int(color_radius),(255,0,255),2)
            # move robot forward and steer proportional target's x-distance from center
            center = (170 - color_x)/170

            #Steering gain PID adjustment
            turn_gain_pid.SystemOutput = center
            turn_gain_pid.SetStepSignal(0)
            turn_gain_pid.SetInertiaTime(0.2, 0.1)

            #Limit the steering gain to the valid range
            target_value_turn_gain = 0.15 + abs(turn_gain_pid.SystemOutput)
            if target_value_turn_gain < 0:
                target_value_turn_gain = 0
            elif target_value_turn_gain > 2:
                target_value_turn_gain = 2

            #Keep the output motor speed within the valid driving range
            target_value_speedl = speed_widget.value - target_value_turn_gain * center
            target_value_speedr = speed_widget.value + target_value_turn_gain * center
            if target_value_speedl<0.3:
                target_value_speedl=0
            elif target_value_speedl>1:
                target_value_speedl = 1
            if target_value_speedr<0.3:
                target_value_speedr=0
            elif target_value_speedr>1:
                target_value_speedr = 1

            robot.set_motors(target_value_speedl, target_value_speedr)
    # No target detected
    else:
        robot.stop()
       
    # Update image display to widget
    image_widget.value = bgr8_to_jpeg(frame)
   
execute({'new': camera.value})
camera.unobserve_all()
camera.observe(execute, names='value')

import time
camera.unobserve_all()
time.sleep(1.0)
robot.stop()
