import os
import sys
import numpy
import time
import pickle
import scipy.ndimage

sys.path.append("/scratch/georgioutk/cliffordConvolutionTurbine/")
import tensorflow as tf
import cliffordConvolution as cc

mnist = tf.keras.datasets.mnist
MOVING_AVERAGE_DECAY = 0.0
checkpoint_dir = "trained_testNewAngles/ccCcFCNet5L3x96_64_2x36_32_16_fs_9_all_7_16Bins_weightMask_trainAll_onlyRescaleBNFC_dropout_3x0.5_0MAD_MaxConv0_droppingLR_30_signedDiffMinus_notMaybe_1Release/"
# checkpoint_dir = "trained_angles_complexWeights/grGrFCNet5L3x96_64_2x36_32_16_fs_9_all_7_16Bins_weightMask_trainAll_dropout_0.2_0MAD_droppingLR_30_1Release/"
# checkpoint_dir = "trained_angles/opOpFCNet5L3x96_64_2x36_32_16_fs_9_all_7_16Bins_weightMask_trainAll_dropout_0.2_0MAD_droppingLR_30_1Release/"
# checkpoint_dir = "trained_complexWeights/grCcFCNet5L3x96_64_2x36_32_16_fs_9_all_7_16Bins_weightMask_trainAll_onlyRescaleBNFC_dropout_3x0.5_0MAD_droppingLR_30_1Release/"
batch_size = 200
numGpus = 1
# import grEqNoVecCnnGrFC_sumCc as network
# import grEqNoVecCnnGrFC as network
# import opCnnOpFCTurbine as network
import ccCnnCcFCTurbine as network
import preprocessing

mnsitTrain = {'x': numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_imanges.npy"), \
'y': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_labels.npy")[:,:1]).astype(numpy.int32), \
'a': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_labels.npy")[:,1:2])}
mnsitTest = {'x': numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_imanges.npy"), \
'y': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_labels.npy")[:,:1]).astype(numpy.int32), \
'a': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_labels.npy")[:,1:2])}

# [next_example, next_label, next_angle], [next_testExample, next_testLabel, next_testAngle], [trainIterator, testIterator] = \
# preprocessing.colorWithRotation(batch_size, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'])
[next_example, next_label, next_angle], [next_testExample, next_testLabel, next_testAngle], [trainIterator, testIterator] = \
preprocessing.gradientOrientationWithAngle(batch_size, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'])

# ims = mnsitTrain['x'][:batch_size]
# next_testExample = tf.constant(ims, dtype=tf.float32)
# next_testExample = cc.layers.calculateImageGradients(next_testExample)

lr = 1
tf.add_to_collection("learning_rate", lr)

with tf.device("/gpu:0"):
	netOut = network.inference(next_testExample, batch_size, "test", first=True, resuse_batch_norm=False, fs=7, normalizationMode="bn")


variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

myconfig = tf.ConfigProto()
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)
writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
	# Restores from checkpoint
	print("Model path:\n{}".format(ckpt.model_checkpoint_path))
	saver.restore(sess, ckpt.model_checkpoint_path)

inputPlaceholders = tf.get_collection("inputTestData")
sess.run(testIterator.initializer, feed_dict={inputPlaceholders[0]: mnsitTest['x'], inputPlaceholders[1]: mnsitTest['y'], inputPlaceholders[2]: mnsitTest['a']})
overallTime = 0
allTimes = []
activations = []
for i in range(101):
	startTime = time.time()
	res = sess.run(netOut)
	activations.append(res)
	endTime = time.time()
	if i != 0:
		overallTime += endTime - startTime
		allTimes.append(endTime - startTime)
	print(i, end='\r', flush=True)

overallTime/100
numpy.save("activationsTestNew", activations)
old = numpy.load("activationsTestOld.npy")
numpy.sum(numpy.array(activations) != old)
