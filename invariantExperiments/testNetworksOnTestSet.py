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

numGpus=1
batch_size = 100
MOVING_AVERAGE_DECAY = 0.99
num_angles = 7
# checkpoint_dir = "/tank/georgioutk/cifar/gor/7LplainBlur/"

release_dir = sys.argv[1]
log = open(sys.argv[1]+"TestAccV16.txt", "w", 1)
# import plainCnn4L as network
# import grEqNoVecCnn55L as network
# import opCnnAllcc55L as network
# import ccCnnlop55L as network
# import ccCnnCcFC as network
# import ccCnnAllcc55L as network
import grSumCcNoVecCnn55L as network
# import cclopCnnAllcc6L as network
# import rotEqNet2 as network
import preprocessing

def testNetwork(sess, top_1, testBatch_size, iterator, gx_test, y_test):
	inputPlaceholders = tf.get_collection("inputTestData")
	sess.run(iterator.initializer, feed_dict={inputPlaceholders[0]: gx_test, inputPlaceholders[1]: y_test})
	correct = 0
	count = 0
	while True:
		try:
			res = sess.run(top_1)
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

# mnsitTrain = numpy.load("/scratch/georgioutk/mnist/mnist_rotation_new/rotated_train.npz")
# mnsitTest = numpy.load("/scratch/georgioutk/mnist/mnist_rotation_new/rotated_test.npz")
mnsitTrainVal = numpy.loadtxt("data/mnist_all_rotation_normalized_float_train_valid.amat", dtype=numpy.float32)
mnsitTestRaw = numpy.loadtxt("data/mnist_all_rotation_normalized_float_test.amat", dtype=numpy.float32)
mnsitTrain = {}
mnsitTest = {}
mnsitTrain['x'] = mnsitTrainVal[:10000,:-1]
mnsitTrain['y'] = mnsitTrainVal[:10000,-1].astype(numpy.int32)
# mnsitTest['x'] = mnsitTrainVal[:,:-1]
# mnsitTest['y'] = mnsitTrainVal[:,-1].astype(numpy.int32)
mnsitTest['x'] = mnsitTestRaw[:,:-1]
mnsitTest['y'] = mnsitTestRaw[:,-1].astype(numpy.int32)

# [next_example, next_label], [next_testExample, next_testLabel], [trainIterator, testIterator] = preprocessing.gradientOrientation(batch_size, mnsitTrain['x'].reshape([10000,28,28,1]), mnsitTrain['y'].reshape([10000,1]))
[next_example, next_label], [next_testExample, next_testLabel], [trainIterator, testIterator] = preprocessing.colorInput(batch_size, mnsitTrain['x'].reshape([10000,28,28,1]), mnsitTrain['y'].reshape([10000,1]))
# [x_trainNval, y_trainNval], [x_test, y_test] = preprocessing.loadMnistData()
# paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
# next_example = tf.pad(next_example, paddings)
# next_testExample = tf.pad(next_testExample, paddings)

rotImages = []
# for ang in range(num_angles):
# 	rotImages.append(tf.contrib.image.rotate(next_testExample, ang*numpy.pi/(2*num_angles), interpolation="BILINEAR"))
lr = 1
tf.add_to_collection("learning_rate", lr)
for ang in range(num_angles):
	# if next_testExample.get_shape()[-1].value == 1:
	# 	next_testExample = tf.contrib.image.rotate(next_testExample, ang*numpy.pi/(2*num_angles), interpolation="BILINEAR")
	# 	blurKernel = tf.stop_gradient(cc.ops.getGuessKernel(1))
	# 	paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
	# 	next_testExample = tf.pad(next_testExample, paddings)
	# 	next_testExample = tf.nn.conv2d(next_testExample, blurKernel, [1,1,1,1], "SAME")
	# 	next_testExample = cc.layers.calculateImageGradients(next_testExample)
	# 	rotImages.append(next_testExample)
	# else:
	# rotImages.append(cc.transformations.rotateVectorField(next_testExample, ang*numpy.pi/(2*num_angles), irelevantAxisFirst=True))
	rotImages.append(tf.contrib.image.rotate(next_testExample, ang*numpy.pi/(2*num_angles), interpolation="BILINEAR"))


# test_nex = tf.split(next_testExample, numGpus, axis=0)

netOut = []
# for i in range(numGpus):
for i in range(num_angles):
	with tf.name_scope('tower_%d' % (i%numGpus)) as scope:
		with tf.device('/gpu:%d' % (i%numGpus)):
			testSoftmax, testProb, skoupidia = network.inference(rotImages[i], batch_size, "test", first=(i==0), resuse_batch_norm=(i!=0), fs=7, normalizationMode="bn")
			netOut.append(testProb)

mean_prob = tf.reduce_mean(tf.stack(netOut), axis=0)

# top_1 = tf.nn.in_top_k(tf.concat(netOut, axis=0), next_testLabel, 1)
top_1 = tf.nn.in_top_k(netOut[0], next_testLabel, 1)
top_1_mean = tf.nn.in_top_k(mean_prob, next_testLabel, 1)

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

SuccRate = None
# SuccRate_summary = tf.Summary()
# SuccRate_summary.value.add(tag='testRotation_error', simple_value=SuccRate)
duration = 0
st = time.time()
SuccRate = testNetwork(sess, top_1, batch_size, testIterator, mnsitTest['x'].reshape([50000,28,28,1]), mnsitTest['y'].reshape([50000,1]))
# SuccRate = testNetwork(sess, top_1, batch_size, testIterator, mnsitTest['x'].reshape([12000,28,28,1]), mnsitTest['y'].reshape([12000,1]))
print("Accuracy: "+str(SuccRate))
# SuccRateMean = testNetwork(sess, top_1_mean, batch_size, testIterator, mnsitTest['x'].reshape([12000,28,28,1]), mnsitTest['y'].reshape([12000,1]))
SuccRateMean = testNetwork(sess, top_1_mean, batch_size, testIterator, mnsitTest['x'].reshape([50000,28,28,1]), mnsitTest['y'].reshape([50000,1]))
print("Mean Accuracy: "+str(SuccRateMean))
print("Accuracy: "+str(SuccRate), file=log)
print("Mean Accuracy: "+str(SuccRateMean), file=log)
