import numpy
import scipy.io
import os
import matplotlib.pyplot as plt
from keras import backend as K




from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np

def VGG_16(weights_path=None):

	K.set_image_dim_ordering('th')

	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model







# Put it all into 2 matrices (one contains pics, one contains ages)

def train():
	model = VGG_16('vgg16_weights.h5')
	print("Got the model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	print("Got the sgd!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	print("Compiled the model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

	#Train model
	data = getData()
	print("Got the data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

	model.fit(data[0], data[1])#, nb_epochs=10, batch_size=10)
	print("Fit the model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

	return (model, data[0], data[1])


#NUM_FOLDERS = 1
PATH = "224_faces"
WIDTH = 224
HEIGHT = 224



def getData():

	directory = PATH #TODO: When multiple folders, loop through each

	numPics = len([name for name in os.listdir(directory)])

	#Create empty numpy array
	data = numpy.empty(shape=(numPics, 3, WIDTH, HEIGHT)) #TODO: Standard amt?

	#Create empty age array
	ages = numpy.empty(shape=(numPics, 1000)) #TODO: IS this the right way to do it?


	#for i in range(NUM_FOLDERS): #Loop through folders
	#	directory = PATH + str(i)
	

	#Loop through files
	i=0
	badPics=0
	for filename in os.listdir(directory):

		#Filename format: 42691_1901-11-03_1974.jpg
		filename2=filename.replace('_','-')
		filename2=filename2.replace('.','-')
		fileWords = filename2.split('-')
		age = calcAge(fileWords[3], fileWords[4], fileWords[2], fileWords[1])
		picData = plt.imread(directory + '/' + filename)

		try :

			# Add another row to our data array
			data[i] = np.swapaxes(picData, 0, 2)
			ages[i,0] = age
			i += 1
			print("good pic")
		except:
			print ("aaa another weird pic")
			badPics += 1

	# Delete empty rows
	badRows = [numPics - 1 - x for x in range(badPics)]
	data = numpy.delete(data, badRows, axis=0)
	ages = numpy.delete(ages, badRows, axis=0)
		
	print("AGES: ")
	print(ages)
	return (data, ages)
	


def testFiles():
	directory = PATH

	#Loop through files
	for filename in os.listdir(directory):

		#Filename format: 42691_1901-11-03_1974.jpg
		filename2=filename.replace('_','-')
		filename2=filename2.replace('.','-')
		fileWords = filename2.split('-')
		age = calcAge(fileWords[3], fileWords[4], fileWords[2], fileWords[1])
		picData = plt.imread(directory + '/' + filename)





def calcAge(currMonth, currYear, birthMonth, birthYear):
	age = int(currYear) - int(birthYear)
	if (int(currMonth) < int(birthMonth)):
		age -= 1
	return age

def getPic():
	return 
		



if __name__ == "__main__":

	#Train model
	trainResult = train()
	model = trainResult[0]
	data = trainResult[1]
	ages = trainResult[2]
	
	# Test trained model
	result=model.evaluate(data, ages)
	print("RESULT")
	print(result)
	#print("\n%s: %.2f%%" % (model.metrics_names[1], result[1]*100))


	for filename in os.listdir(PATH):

		try:
			im = plt.imread(PATH + '/' + filename)
			out = model.predict(im)
			print("PREDICTION")
			print np.argmax(out)
		except:
			pass

	np.savetext('dataTry2.out', )

