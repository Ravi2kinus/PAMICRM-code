from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path
from tensorflow.keras import models
import cv2
from os import listdir
from os.path import isfile, join
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import EHO as eho

model_name = "VGGNet2.model"
rows = 128
cols = 128
channels = 3

if(path.exists(model_name)) :
    model = keras.models.load_model(model_name)
else :
    train_dir = 'dataset/train'
    test_dir = 'dataset/train'
    validation_dir = 'dataset/train'
    
    try :
        train_dir = eho.applyDEHO(train_dir)
    except :
        train_dir = 'dataset/train'
        
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    train_batch_size = 16;
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    
    #Used to get the same results everytime
    np.random.seed(42)
    #tf.random.set_seed(42)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(rows,cols),
        batch_size=train_batch_size,
        class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(rows,cols),
        batch_size=20,
        class_mode='categorical')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(rows,cols),
        batch_size=20,
        class_mode='categorical')
    
    ######################################################################
    #initialize the NN
    
    #Load the VGG16 model, use the ILSVRC competition's weights
    #include_top = False, means only include the Convolution Base (do not import the top layers or NN Layers)
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(rows,cols,channels))
    conv_base.trainable = False;
    model = models.Sequential()
    
    #Add the VGGNet model
    model.add(conv_base)
    
    #NN Layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation='relu'))
    model.add(keras.layers.Dense(3,activation='softmax'))
    
    print(model.summary())
    ######################################################################
    
    #Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    #Steps per epoch = Number of images in the training directory / batch_size (of the generator)
    #validation_steps = Number of images in the validation directory / batch_size (of the generator)
    checkpoint_callback = keras.callbacks.ModelCheckpoint("%s" % (model_name), save_best_only=True)
    model_history = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs = 10,
        validation_data=validation_generator,
        validation_steps=20,
        callbacks = [checkpoint_callback])
    
    #Plot the model
    pd.DataFrame(model_history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
    model.save(model_name)
    
    #Save the history in CSV file
    hist_csv_file = 'vggnet_history.csv'
    hist_df = pd.DataFrame(model_history.history)
    with open(hist_csv_file,mode='w') as f:
        hist_df.to_csv(f)

#Check the testing set
#model.evaluate_generator(test_generator,steps=50)
folderName = "dataset/test";
onlyfiles = [f for f in listdir(folderName) if isfile(join(folderName, f))]
out_csv = 'output.csv'
output = 'Frame, Obtained class'

for count in range(0,len(onlyfiles)):
    print(("Processing:%s\\%s\n" % (folderName, onlyfiles[count])))
    
    frame = cv2.imread(("%s\\%s" % (folderName, onlyfiles[count])),cv2.IMREAD_UNCHANGED)
    frame = cv2.resize(frame, (rows,cols), interpolation = cv2.INTER_AREA)
    frame_bkp = np.zeros(shape = (rows,cols,channels))
    try :
        frame_bkp[:,:,0] = frame
        frame_bkp[:,:,1] = frame
        frame_bkp[:,:,2] = frame
    except :
        frame_bkp = frame
        print('Dimensions done!')
    
    frame = np.asarray(frame_bkp).reshape((1,rows,cols,channels))
    y_pred = model.predict_classes(frame)
    y_pred = y_pred[0]
    if(y_pred == 0) :
        label = 'Type 1'
    elif(y_pred == 1) :
        label = 'Type 2'
    else :
        label = 'Type 3'
    
    output = output + "\n" + onlyfiles[count] + "," + label
    
    print("Classified as type %s\n" % (label))
    cv2.putText(frame_bkp, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
    cv2.imshow("Result", frame_bkp)
    cv2.waitKey(1000)
cv2.destroyAllWindows()
file1 = open(out_csv,"w") 
file1.write(output)
file1.close()