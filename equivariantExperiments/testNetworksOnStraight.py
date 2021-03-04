import os
import sys
import numpy
import time
import pickle
import scipy.ndimage
from multiprocessing import Pool

sys.path.append("/scratch/georgioutk/cliffordConvolution/")
import tensorflow as tf
import cliffordConvolution as cc

numGpus=3
batch_size = 4000
MOVING_AVERAGE_DECAY = 0.99
num_angles = 32
# checkpoint_dir = "/tank/georgioutk/cifar/gor/7LplainBlur/"

release_dir = sys.argv[1]
# log = open(sys.argv[1]+"TestAccV16Original.txt", "w", 1)
# import opCnnAllcc55L as network
# import ccCnnlop55L as network
# import ccCnnAllcc55L as network
import ccCnnCcFC as network
# import opCnnOpFC as network
# import grEqNoVecCnnGrFC_sumCc as network
# import grEqNoVecCnn55L as network
# import grEqNoVecNoSteerCnn55L as network
import preprocessing

def testNetwork(sess, top_1, testBatch_size, iterator, gx_test, y_test, angle, tfAngle):
	inputPlaceholders = tf.get_collection("inputTestData")
	sess.run(iterator.initializer, feed_dict={inputPlaceholders[0]: gx_test, inputPlaceholders[1]: y_test})
	correct = 0
	count = 0
	while True:
		try:
			res = sess.run(top_1, feed_dict={tfAngle: angle})
			correct += numpy.sum(res)
			count += testBatch_size
			if count % 100 == 0:
				print("%2.2f"%(count*100/gx_test.shape[0]), end="\r", flush=True)
		except tf.errors.OutOfRangeError:
			break
	return correct/count

def rotateDataset(data):
	dataset = data[0]
	res = numpy.zeros(dataset.shape)
	for i in range(dataset.shape[0]):
		if data[1] == None:
			angle = numpy.random.rand()*2*numpy.pi
		else:
			angle = data[1]
		res[i] = scipy.ndimage.rotate(dataset[i], numpy.degrees(angle), order=3, reshape=False)
	return res

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = numpy.expand_dims((x_train / 255.0).astype(numpy.float32), -1), numpy.expand_dims((x_test / 255.0).astype(numpy.float32), -1)
y_train, y_test = numpy.expand_dims(y_train.astype(numpy.int32), -1), numpy.expand_dims(y_test.astype(numpy.int32), -1)
trainInds = numpy.load("uprightInds.npy")
allInds = numpy.arange(x_train.shape[0])
testInds = numpy.setxor1d(allInds, trainInds)
mnsitTrain = {'x': x_train[trainInds], 'y': y_train[trainInds], 'a': numpy.zeros([trainInds.shape[0],1], dtype=numpy.float32)}
mnsitTest = {'x': x_train[testInds], 'y': y_train[testInds], 'a': numpy.zeros([testInds.shape[0],1], dtype=numpy.float32)}

[next_example, next_label], [next_testExample, next_testLabel], [trainIterator, testIterator] = preprocessing.gradientOrientation(batch_size, mnsitTrain['x'], mnsitTrain['y'])
# [next_example, next_label], [next_testExample, next_testLabel], [trainIterator, testIterator] = preprocessing.colorInput(batch_size, mnsitTrain['x'], mnsitTrain['y'])
# [x_trainNval, y_trainNval], [x_test, y_test] = preprocessing.loadMnistData()
# paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
# next_example = tf.pad(next_example, paddings)
# next_testExample = tf.pad(next_testExample, paddings)

ang = tf.placeholder(tf.float32, [])
rotImages = cc.transformations.rotateVectorField(next_testExample, ang, irelevantAxisFirst=True)
# rotImages = tf.contrib.image.rotate(next_testExample, ang, interpolation="BILINEAR")


# test_nex = tf.split(next_testExample, numGpus, axis=0)
perGPUImages = tf.split(rotImages, 4, axis=0)
lr = 1
tf.add_to_collection("learning_rate", lr)
with tf.device('/gpu:0'):
	testSoftmax, testProb, skoupidia = network.inference(perGPUImages[0], batch_size/2, "test", first=True, resuse_batch_norm=False, fs=7, normalizationMode="bn")
	# testSoftmax, testProb, skoupidia = network.inference(rotImages, batch_size, "test", first=True, resuse_batch_norm=False, fs=7, normalizationMode="bn")

with tf.device('/gpu:1'):
	testSoftmax2, testProb2, skoupidia2 = network.inference(perGPUImages[1], batch_size/2, "test", first=False, resuse_batch_norm=True, fs=7, normalizationMode="bn")
	# testSoftmax, testProb, skoupidia2, skoupidia = network.inference(rotImages, batch_size, "test", first=True, resuse_batch_norm=False, fs=7, normalizationMode="bn")

with tf.device('/gpu:2'):
	testSoftmax3, testProb3, skoupidia3 = network.inference(perGPUImages[2], batch_size/2, "test", first=False, resuse_batch_norm=True, fs=7, normalizationMode="bn")
	# testSoftmax, testProb, skoupidia2, skoupidia = network.inference(rotImages, batch_size, "test", first=True, resuse_batch_norm=False, fs=7, normalizationMode="bn")

with tf.device('/gpu:3'):
	testSoftmax4, testProb4, skoupidia4 = network.inference(perGPUImages[2], batch_size/2, "test", first=False, resuse_batch_norm=True, fs=7, normalizationMode="bn")
	# testSoftmax, testProb, skoupidia2, skoupidia = network.inference(rotImages, batch_size, "test", first=True, resuse_batch_norm=False, fs=7, normalizationMode="bn")


testProb = tf.concat([testProb, testProb2, testProb3, testProb4], axis=0)
# top_1 = tf.nn.in_top_k(tf.concat(netOut, axis=0), next_testLabel, 1)
top_1 = tf.nn.in_top_k(testProb, next_testLabel, 1)

variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()

saver = tf.train.Saver(variables_to_restore)

myconfig = tf.ConfigProto()
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)
writer = tf.summary.FileWriter(release_dir, sess.graph)
ckpt = tf.train.get_checkpoint_state(release_dir)
if ckpt and ckpt.model_checkpoint_path:
	# Restores from checkpoint
	print("Model path:\n{}".format(ckpt.model_checkpoint_path))
	saver.restore(sess, ckpt.model_checkpoint_path)

SuccRate = []
for an in range(num_angles):
	print("Angle: "+str(an/num_angles))
	angle = an*2*numpy.pi/num_angles
	SuccRate.append(testNetwork(sess, top_1, batch_size, testIterator, mnsitTest['x'], mnsitTest['y'], angle, ang))

SuccRate = numpy.array(SuccRate)
numpy.save(release_dir+"perAnSucRate32", SuccRate)