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
# log = open(sys.argv[1]+"TestAccV16_08.txt", "w", 1)
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

[next_example, next_label, next_angle], [next_testExample, next_testLabel, next_testAngle], [trainIterator, testIterator] = \
preprocessing.colorWithRotation(batch_size, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'])
# [next_example, next_label, next_angle], [next_testExample, next_testLabel, next_testAngle], [trainIterator, testIterator] = \
# preprocessing.gradientOrientationWithAngle(batch_size, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'])

rotImages = []
# for ang in range(num_angles):
# 	rotImages.append(tf.contrib.image.rotate(next_testExample, ang*numpy.pi/(2*num_angles), interpolation="BILINEAR"))
lr = 1
tf.add_to_collection("learning_rate", lr)
netOut = []
netAngles = []
# for i in range(numGpus):
with tf.name_scope('tower_0') as scope:
	with tf.device('/gpu:0'):
		testSoftmax, testProb, pred_angles = network.inference(next_testExample, batch_size, "test", first=True, resuse_batch_norm=False, fs=7, normalizationMode="bn")
		testPredAngle = tf.squeeze(cc.ops.reduceIndex(pred_angles, tf.expand_dims(tf.expand_dims(next_testLabel, axis=-1), axis=-1)))

anDif = next_testAngle - testPredAngle
# testAnDif = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(anDif), tf.cos(anDif))))
# print(testAnDif.get_shape())
# exit()
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

inputPlaceholders = tf.get_collection("inputTestData")
sess.run(testIterator.initializer, feed_dict={inputPlaceholders[0]: mnsitTrain['x'], inputPlaceholders[1]: mnsitTrain['y'], inputPlaceholders[2]: mnsitTrain['a']})
outLabels = []
angDif = []
count = 0
while True:
	try:
		res = sess.run([next_testLabel, anDif])
		outLabels.append(res[0])
		angDif.append(res[1])
		if count % 100 == 0:
			print("%2.2f"%(count*batch_size/mnsitTrain['x'].shape[0]), end="\r", flush=True)
		count += 1
	except tf.errors.OutOfRangeError:
		break

outLabels = numpy.concatenate(outLabels, axis=0)
angDif = numpy.concatenate(angDif, axis=0)
# numpy.save(release_dir+"exampleLabels", outLabels)
# numpy.save(release_dir+"exampleAngleDifs", angDif)

angSin = numpy.sin(angDif)
angCos = numpy.cos(angDif)

offset = numpy.zeros(10, dtype=numpy.float32)
for a in range(10):
	inds = numpy.where(outLabels==a)
	offset[a] = numpy.math.atan2(numpy.sum(angSin[inds]), numpy.sum(angCos[inds]))

numpy.save(release_dir+"angleOffsets_16", offset)