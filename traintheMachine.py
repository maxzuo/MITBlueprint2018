import numpy as np
import random
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import os
import datetime

hand = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
hand1 = cv2.imread("test1.png", cv2.IMREAD_GRAYSCALE)
hand2 = cv2.imread("test2.png", cv2.IMREAD_GRAYSCALE)

# while True:
# 	cv2.imshow("4", hand)
# 	cv2.imshow("2", hand1)

# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break


# cap.release()
# cv2.destroyAllWindows()

class NeuronLayer():

	def __init__(self, size, thetaSize):
		self.synapticWeights = 2 * np.random.rand(thetaSize, size) - 1

class NeuralNetwork():

	def __init__(self, layer1, layer2):

		self.layer1 = layer1
		self.layer2 = layer2

	def __str__(self):
		return "Layer one weights:\n" + str(self.layer1.synapticWeights) + "\n\nLayer two weights:\n" + str(self.layer2.synapticWeights)

	def sigmoid(self, x, deriv = False):
		if deriv:
			return x * (1 - x)
		return 1 / (1 + np.exp(-x))

	def train(self, trainingSetInputs, trainingSetOutputs, iterations):

		for iteration in xrange(iterations):

			#Compute the result first
			layer1Output, layer2Output = self.compute(trainingSetInputs)

			#Begin backpropagation
			layer2Error = trainingSetOutputs - layer2Output
			layer2Delta = layer2Error * self.sigmoid(layer2Output, deriv=True)

			layer1Error = layer2Delta.dot(self.layer2.synapticWeights.T)
			layer1Delta = layer1Error * self.sigmoid(layer1Output, deriv=True)

			#Create weight adjustments

			layer1Adjustment = trainingSetInputs.T.dot(layer1Delta)
			layer2Adjustment = layer1Output.T.dot(layer2Delta)

			#Actually adjust weights

			self.layer1.synapticWeights += layer1Adjustment
			self.layer2.synapticWeights += layer2Adjustment


	def compute(self, inputs):
		
		layer1Output = self.sigmoid(np.dot(inputs, self.layer1.synapticWeights))
		layer2Output = self.sigmoid(np.dot(layer1Output, self.layer2.synapticWeights))
		
		return layer1Output, layer2Output



if __name__ == "__main__":

	#Seed random in order to make debugging easier
	layer1 = NeuronLayer(15, 1024)
	layer2 = NeuronLayer(3, 15)

	#create the neural network.
	network = NeuralNetwork(layer1, layer2)

	# trainingSetInputs = np.array([hand.flatten(), hand1.flatten(), hand2.flatten()])
	# trainingSetOutputs = np.array()

	print network
	startTime = datetime.datetime.now()
	network.train(trainingSetInputs, trainingSetOutputs, 200000)
	endTime = datetime.datetime.now()

	timeDelta = endTime - startTime
	print "Time elapsed: ", timeDelta

	print "\n\n\n\n\n\n\n...\n\n\n\n\n\n\n"
	print network

	print "\n\n\n\n\n\n\n...\n\n\n\n\n\n\n"
	print "Time elapsed: ", timeDelta
	_, output = network.compute((cv2.imread("test2.png", cv2.IMREAD_GRAYSCALE).flatten()))
	# print "test2.png: ", output
	# if output[1] > output[0]:
	# 	print "2 fingers, confidence:", str(int(100 * (output[1] - output[0]))) + "%"
	# else:
	# 	print "4 fingers, confidence:", str(int(100 * (output[0] - output[1]))) + "%"

	print output

	# _, output = network.compute(
	# print "Hand2: ", output


