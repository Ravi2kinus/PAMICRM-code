from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from os import path
import cv2
import easygui as eg
import sys
import matplotlib.image as mpimg

model_name = "VGGNet.model"
rows = 128
cols = 128
channels = 3

if(path.exists(model_name)) :
    model = keras.models.load_model(model_name)
else :
    print('Unable to load model...')
    sys.exit(0)


file = eg.fileopenbox('Pick an image file')

frame = cv2.imread(file,cv2.IMREAD_UNCHANGED)
frame = cv2.resize(frame, (rows,cols), interpolation = cv2.INTER_AREA)
frame_bkp = np.zeros(shape = (rows,cols,channels))
try :
    frame_bkp[:,:,0] = frame
    frame_bkp[:,:,1] = frame
    frame_bkp[:,:,2] = frame
except :
    frame_bkp = frame[:,:,0:3]
    print('Dimensions done!')

frame = np.asarray(frame_bkp).reshape((1,rows,cols,channels))
y_pred = model.predict_classes(frame)
y_pred = y_pred[0]
label = str(y_pred)

print("Classified as type %s\n" % (label))
#cv2.putText(frame_bkp, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
#cv2.imshow("Result", frame_bkp)
cv2.destroyAllWindows()

outFrame = frame_bkp
rows = len(frame_bkp)
cols = len(frame_bkp[0])

strVal = 'Red: Land, Green: Forest, Blue: Water, RG:Barren, Green Blue: Urban'
areaLand = 0
areaForest = 0
areaWater = 0
areaBarren = 0
areaUrban = 0

for row in range(0, rows) :
    for col in range(0, cols) :
        intensity = frame_bkp[row, col, 0]
        
        if(intensity < 25) :
            #Low intensity land regions
            red = 255
            green = 0
            blue = 0
            areaLand = areaLand + 1
            
        elif(intensity < 60) :
            #Moderate intensity regions
            red = 0
            green = 255
            blue = 0
            
            areaForest = areaForest + 1
        elif(intensity < 110) :
            #High intensity water regions
            red = 0
            green = 0
            blue = 255
            
            areaWater = areaWater + 1
        elif(intensity < 120) :
            #Barren
            red = 0
            green = 255
            blue = 255
            
            areaBarren = areaBarren + 1
        elif(intensity < 150) :
            #Barren
            red = 128
            green = 255
            blue = 128
            
            areaBarren = areaBarren + 1
        else :
            #Urban
            red = 0
            green = 255
            blue = 255
            
            areaUrban = areaUrban + 1
            
        outFrame[row, col, 0] = red
        outFrame[row, col, 1] = green
        outFrame[row, col, 2] = blue


imgPath = file
img = mpimg.imread(imgPath)
imgplot = plt.imshow(img)
plt.show()

imgPath = 'out_img.png'
cv2.imwrite(imgPath, outFrame)
img = mpimg.imread(imgPath)
imgplot = plt.imshow(img)
plt.title(strVal)

plt.show()

areaLand = areaLand * 100 / (rows*cols)
areaForest = areaForest * 100 / (rows*cols)
areaWater = areaWater * 100 / (rows*cols)
areaBarren = areaBarren * 100 / (rows*cols)
areaUrban = areaUrban * 100 / (rows*cols)

drinkable = 100*areaWater / (areaWater+areaBarren)
plantable = 100*(areaLand+areaForest)/(areaLand+areaForest+areaWater+areaBarren+areaUrban)

print('Land Area %0.04f acres' % areaLand)
print('Forest Area %0.04f acres' % areaForest)
print('Water Area %0.04f acres' % areaWater)
print('Barren Area %0.04f acres' % areaBarren)
print('Urban Area %0.04f acres' % areaUrban)

print('******************************************')
print('QUALITY ANALYSIS')
print('******************************************')
print('Water in this area is %0.04f %% drinkable' % (drinkable))
print('Land in this area is %0.04f %% plantable' % (plantable))