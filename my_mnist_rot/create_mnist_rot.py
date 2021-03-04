import numpy
import tensorflow as tf
import scipy.ndimage

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = numpy.expand_dims((x_train / 255.0).astype(numpy.float32), -1), numpy.expand_dims((x_test / 255.0).astype(numpy.float32), -1)
y_train, y_test = numpy.expand_dims(y_train.astype(numpy.int32), -1), numpy.expand_dims(y_test.astype(numpy.int32), -1)

x = numpy.concatenate([x_train, x_test], axis=0)
y = numpy.concatenate([y_train, y_test], axis=0)

numTrainExamples = 12000
numTestExamples = 50000

trainSet = numpy.zeros([numTrainExamples, 28, 28, 1], dtype=numpy.float32)
testSet = numpy.zeros([numTestExamples, 28, 28, 1], dtype=numpy.float32)
trainLabels = numpy.zeros([numTrainExamples, 2], dtype=numpy.float32)
testLabels = numpy.zeros([numTestExamples, 2], dtype=numpy.float32)
for i in range(numTrainExamples):
	a = numpy.random.rand()*2*numpy.pi
	im = int(numpy.random.rand()*x.shape[0])
	trainSet[i] = scipy.ndimage.rotate(x[im], numpy.degrees(a), order=3, reshape=False)
	trainLabels[i, 0] = y[im].astype(numpy.float32)
	trainLabels[i, 1] = a

for i in range(numTestExamples):
	a = numpy.random.rand()*2*numpy.pi
	im = int(numpy.random.rand()*x.shape[0])
	testSet[i] = scipy.ndimage.rotate(x[im], numpy.degrees(a), order=3, reshape=False)
	testLabels[i, 0] = y[im].astype(numpy.float32)
	testLabels[i, 1] = a

numpy.save("mnist_rot_train_imanges", trainSet)
numpy.save("mnist_rot_train_labels", trainLabels)
numpy.save("mnist_rot_test_imanges", testSet)
numpy.save("mnist_rot_test_labels", testLabels)
