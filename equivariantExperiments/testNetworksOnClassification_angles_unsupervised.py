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
log = open(sys.argv[1]+"TestAccV16WithAngle.txt", "w", 1)
# import opCnnAllcc55L as network
# import ccCnnlop55L as network
# import ccCnnAllcc55L as network
import grEqNoVecCnnGrFC_sumCc as network
# import ccCnnCcFC as network
# import opCnnOpFC as network
import preprocessing

def testNetwork(sess, top_1, testBatch_size, iterator, gx_test, y_test, a_test):
	inputPlaceholders = tf.get_collection("inputTestData")
	sess.run(iterator.initializer, feed_dict={inputPlaceholders[0]: gx_test, inputPlaceholders[1]: y_test, inputPlaceholders[2]: a_test})
	correct = 0
	anDif = 0
	count = 0
	while True:
		try:
			res = sess.run(top_1)
			correct += numpy.sum(res[0])
			anDif += res[1]
			count += 1
			if count % 100 == 0:
				print("%2.2f"%(count*100/gx_test.shape[0]), end="\r", flush=True)
		except tf.errors.OutOfRangeError:
			break
	# print("Validation accuracy: "+str(correct/count), file=log)
	# return correct/count
	return correct/(count*testBatch_size), anDif/count

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
# mnsitTrainVal = numpy.loadtxt("data/mnist_all_rotation_normalized_float_train_valid.amat", dtype=numpy.float32)
# mnsitTestRaw = numpy.loadtxt("data/mnist_all_rotation_normalized_float_test.amat", dtype=numpy.float32)
# mnsitTrain = {}
# mnsitTest = {}
# mnsitTrain['x'] = mnsitTrainVal[:10000,:-1]
# mnsitTrain['y'] = mnsitTrainVal[:10000,-1].astype(numpy.int32)
# mnsitTest['x'] = mnsitTrainVal[:,:-1]
# mnsitTest['y'] = mnsitTrainVal[:,-1].astype(numpy.int32)
# mnsitTest['x'] = mnsitTestRaw[:,:-1]
# mnsitTest['y'] = mnsitTestRaw[:,-1].astype(numpy.int32)

mnsitTrain = {'x': numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_imanges.npy"), \
'y': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_labels.npy")[:,:1]).astype(numpy.int32), \
'a': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_labels.npy")[:,1:2])}
mnsitTest = {'x': numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_imanges.npy"), \
'y': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_labels.npy")[:,:1]).astype(numpy.int32), \
'a': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_labels.npy")[:,1:2])}

# [next_example, next_label, next_angle], [next_testExample, next_testLabel, next_testAngle], [trainIterator, testIterator] = \
# preprocessing.gradientOrientationWithAngle(batch_size, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'])
[next_example, next_label, next_angle], [next_testExample, next_testLabel, next_testAngle], [trainIterator, testIterator] = \
preprocessing.colorWithRotation(batch_size, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'])

rotImages = []
# for ang in range(num_angles):
# 	rotImages.append(tf.contrib.image.rotate(next_testExample, ang*numpy.pi/(2*num_angles), interpolation="BILINEAR"))
lr = 1
angles = []
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
	angles.append(ang*numpy.pi/(2*num_angles))
	# rotImages.append(cc.transformations.rotateVectorField(next_testExample, angles[-1], irelevantAxisFirst=True))
	rotImages.append(tf.contrib.image.rotate(next_testExample, ang*numpy.pi/(2*num_angles), interpolation="BILINEAR"))


# test_nex = tf.split(next_testExample, numGpus, axis=0)
offsets = numpy.load(release_dir+"angleOffsets_16.npy")
offsets = tf.constant(offsets)
netOut = []
netAngles = []
# for i in range(numGpus):
for i in range(num_angles):
	with tf.name_scope('tower_%d' % (i%numGpus)) as scope:
		with tf.device('/gpu:%d' % (i%numGpus)):
			testSoftmax, testProb, pred_angles = network.inference(rotImages[i], batch_size, "test", first=(i==0), resuse_batch_norm=(i!=0), fs=7, normalizationMode="bn")
			testPredAngle = tf.squeeze(cc.ops.reduceIndex(pred_angles, tf.expand_dims(tf.expand_dims(next_testLabel, axis=-1), axis=-1))) - angles[i]
			perExampleOffsets = tf.gather(offsets, next_testLabel)
			netOut.append(testProb)
			netAngles.append(testPredAngle+perExampleOffsets)


stackedNetAngles = tf.stack(netAngles, axis=0)
cosNetAngle = tf.reduce_mean(tf.cos(stackedNetAngles), axis=0)
sinNetAngle = tf.reduce_mean(tf.sin(stackedNetAngles), axis=0)

mean_angles = tf.atan2(sinNetAngle, cosNetAngle)
mean_prob = tf.reduce_mean(tf.stack(netOut), axis=0)
mean_anDif = next_testAngle - mean_angles
# next_testAngle2 = tf.where(tf.logical_or(tf.equal(next_testLabel, 0), tf.equal(next_testLabel, 8)), next_testAngle-numpy.pi, next_testAngle)
# mean_anDif2 = next_testAngle2 - mean_angles
mean_testAnDif = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(mean_anDif), tf.cos(mean_anDif))))
# mean_testAnDif2 = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(mean_anDif2), tf.cos(mean_anDif2))))
# mean_testAnDif = tf.reduce_min(tf.stack([mean_testAnDif, mean_testAnDif2], axis=0), axis=0)

perAngleDif = []
for i in range(num_angles):
	anDif = next_testAngle - netAngles[i]
	# anDif2 = next_testAngle2 - netAngles[i]
	testAnDif = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(anDif), tf.cos(anDif))))
	# testAnDif2 = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(anDif2), tf.cos(anDif2))))
	# testAnDif = tf.reduce_min(tf.stack([testAnDif, testAnDif2], axis=0), axis=0)
	perAngleDif.append(testAnDif)
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
SuccRate, anErr = testNetwork(sess, [top_1, testAnDif], batch_size, testIterator, mnsitTest['x'], mnsitTest['y'], mnsitTest['a'])
# SuccRate = testNetwork(sess, top_1, batch_size, testIterator, mnsitTest['x'].reshape([12000,28,28,1]), mnsitTest['y'].reshape([12000,1]))
print("Accuracy/AngleError: "+str(SuccRate)+"/"+str(anErr))
# SuccRateMean = testNetwork(sess, top_1_mean, batch_size, testIterator, mnsitTest['x'].reshape([12000,28,28,1]), mnsitTest['y'].reshape([12000,1]))
SuccRateMean, mean_anErr = testNetwork(sess, [top_1_mean, mean_testAnDif], batch_size, testIterator, mnsitTest['x'], mnsitTest['y'], mnsitTest['a'])
print("Mean Accuracy/AngleError: "+str(SuccRateMean)+"/"+str(mean_anErr))
print("Accuracy/AngleError: "+str(SuccRate)+"/"+str(anErr), file=log)
print("Mean Accuracy/AngleError: "+str(SuccRateMean)+"/"+str(mean_anErr), file=log)
# for i in range(num_angles):
# 	SuccRate, anErr = testNetwork(sess, [top_1, perAngleDif[i]], batch_size, testIterator, mnsitTest['x'], mnsitTest['y'], mnsitTest['a'])
# 	# SuccRate = testNetwork(sess, top_1, batch_size, testIterator, mnsitTest['x'].reshape([12000,28,28,1]), mnsitTest['y'].reshape([12000,1]))
# 	print("Angle/AngleError: "+str(angles[i])+"/"+str(anErr))
