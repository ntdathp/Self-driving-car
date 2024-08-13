import cv2
import numpy as np
import random
import pandas as pd
import glob 
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.optimizers import Adam		
import matplotlib.pyplot as plt

IMG_H,IMG_W,IMG_CH=160,320,1
def augment(image, measurement):
	#đảo dữ liệu ngẫu nhiên
	if np.random.rand() < 0.5:
		image = np.fliplr(image)
		measurement[1] = -measurement[1]
		
	return image, measurement

def load_data(test_size):
	#nạp data 
	names = ['center','speed','steering']
	data_df = pd.read_csv('driving_log.csv',header=None,names=names)
	X = data_df['center'].values
	y1= data_df['speed'].values
	y2 = data_df['steering'].values
	yy = np.concatenate(([y1], [y2]), axis = 0)
	y = np.transpose(yy)
	return train_test_split(X,y,test_size=test_size) # chia thành các mẫu test và train :D

def build_model():

	model = Sequential()
	model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(IMG_H,IMG_W,IMG_CH)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))
	model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Dense(2))
	return model

def batch_generator(image_paths,meas,batch_size):
	#lấy batch_size data ngẫu nhiên để train :D làm lại n lần 
	images = np.empty([batch_size,IMG_H,IMG_W,IMG_CH])
	measurements = np.empty([batch_size,2])
	measurement = np.empty(2)
	while True:
		i=0
		for index in np.random.permutation(image_paths.shape[0]):
			center = image_paths[index]
			img = cv2.imread(center)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			image = img.reshape((160, 320, 1))
			measurement[0] = meas[index,0]/60
			measurement[1] = meas[index,1]/25

			image,measurement = augment(image,measurement)
			images[i] = image
			measurements[i] = measurement
			
			i+=1
			if i== batch_size:
				break
		yield images, measurements



def main():
	test_size = 0.2
	batch_size=64
	epochs = 10
	verbose =1
	print('Loading data....')
	X_train,X_val,y_train,y_val = load_data(test_size)
	print('Building models...')
	model = build_model()
	
	print('Compiling model...')
	model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
	print('Training...')
	history_object = model.fit(batch_generator(X_train,y_train,batch_size),
										steps_per_epoch= int(len(X_train)/batch_size),#làm lại bằng số mẫu chia số batch 
										validation_data=batch_generator(X_val,y_val,batch_size),
										validation_steps=int(len(X_val)/batch_size),
										epochs=epochs,
										verbose=verbose)
	print('Saving models...')
	model.save('model.h5')
	print('Model saved')
	print(history_object.history.keys())
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','test'],loc='upper left')
	plt.savefig('history.png',bbox_inches='tight')

if __name__ == '__main__':
	main()

	