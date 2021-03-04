import os
import sys
import numpy
import pickle
import scipy.ndimage
from matplotlib import pyplot as plt

sys.path.append("/scratch/georgioutk/cliffordConvolution/")
import tensorflow as tf
import cliffordConvolution as cc

mnist = tf.keras.datasets.mnist
MOVING_AVERAGE_DECAY = 0.0
checkpoint_dir = "trained_classification/grGr_sum_CcFCNet5L3x96_64_2x36_32_16_fs_9_all_7_16Bins_adjustedBN_weightMask_trainAll_dropout_3x0.5_0MAD_droppingLR_30_1Release/"
num_angles = 16
batch_size = num_angles
numGpus = 1
import grEqNoVecCnnGrFC_sumCc as network
import preprocessing


def plot_field(field):
	fig, axes = plt.subplots(nrows=field.shape[0], ncols=field.shape[1])
	for i in range(field.shape[0]):
		for j in range(field.shape[1]):
			axes[i,j].quiver(0, 0, field[i,j,1], -field[i,j,0], angles='xy', scale_units='xy', scale=1)
			axes[i,j].set_xlim(-1.5, 1.5)
			axes[i,j].set_ylim(-1.5, 1.5)
			axes[i,j].set_xticks([])
			axes[i,j].set_yticks([])
	plt.subplots_adjust(wspace=0, hspace=0)

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

print(next_testExample.get_shape())
rotImages = []
rotImAngle = []
rotImLabel = []
lr = 1
angles = []
tf.add_to_collection("learning_rate", lr)
imPlaceholder = tf.placeholder(tf.float32, [1, 28, 28, 1])
imLabel = tf.placeholder(tf.int32, [1])
imAngle = tf.placeholder(tf.float32, [1])
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
	rotImAngle.append(imAngle+angles[-1])
	rotImLabel.append(imLabel)
	# rotImages.append(cc.transformations.rotateVectorField(imPlaceholder, [angles[-1]], irelevantAxisFirst=True))
	rotImages.append(tf.contrib.image.rotate(imPlaceholder, ang*numpy.pi/(2*num_angles), interpolation="BILINEAR"))

rotImages = tf.concat(rotImages, axis=0)
rotImAngle = tf.concat(rotImAngle, axis=0)
rotImLabel = tf.concat(rotImLabel, axis=0)

# test_nex = tf.split(next_testExample, numGpus, axis=0)

netOut = []
netAngles = []
with tf.device('/gpu:0'):
	testSoftmax, testProb, pred_angles = network.inference(rotImages, batch_size, "test", first=True, resuse_batch_norm=False, fs=7, normalizationMode="bn")
	testPredAngle = tf.squeeze(cc.ops.reduceIndex(pred_angles, tf.expand_dims(tf.expand_dims(rotImLabel, axis=-1), axis=-1)))
	netOut = testProb
	netAngles = testPredAngle


# for i in range(num_angles):
# 	with tf.name_scope('tower_%d' % (i%numGpus)) as scope:
# 		with tf.device('/gpu:%d' % (i%numGpus)):
# 			testSoftmax, testProb, pred_angles = network.inference(rotImages, batch_size, "test", first=(i==0), resuse_batch_norm=(i!=0), fs=7, normalizationMode="bn")
# 			testPredAngle = tf.squeeze(cc.ops.reduceIndex(pred_angles, tf.expand_dims(tf.expand_dims(rotImLabel, axis=-1), axis=-1)))
# 			netOut.append(testProb)
# 			netAngles.append(testPredAngle)


activations = tf.get_collection("activationMagnitudes")
print(len(activations))
print(activations[-1].get_shape())

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


nActivations = sess.run([rotImages, rotImAngle, rotImLabel]+activations, feed_dict={imPlaceholder:mnsitTest['x'][:1], imLabel:mnsitTest['y'][1], imAngle:mnsitTest['a'][1]})
fc2 = numpy.reshape(nActivations[-2], [16,1,1,16,96,2])
plot_field(10*fc2[:,0,0,:,0,:])
plt.show()


weights = tf.get_collection("weights")
actsMagn = tf.get_collection("activationMagnitudes")
actsAngles = tf.get_collection("activationAngles")
acts = tf.get_collection("activations")

# featMapsMagns = []
# featMapsAngs = []
# featMapsCart = []
# featMapRotations = []
# featMapRotPolar = []
# rotationAngle = tf.placeholder(tf.float32, [])
# for i in range(len(actsMagn)):
# 	featMapsMagns.append(tf.placeholder(tf.float32, [numAngles, actsMagn[i].get_shape()[-3], actsMagn[i].get_shape()[-2], 1]))
# 	featMapsAngs.append(tf.placeholder(tf.float32, [numAngles, actsMagn[i].get_shape()[-3], actsMagn[i].get_shape()[-2], 1]))
# 	featMapsCart.append(cc.transformations.changeToCartesian(featMapsMagns[i], featMapsAngs[i]))
# 	# featMapRotations.append(cc.transformations.rotateVectorField(featMapsCart[i], rotationAngle*numpy.pi, irelevantAxisFirst=True))
# 	# featMapRotPolar.append(cc.transformations.changeToPolar(featMapRotations[i], 1))



# poolingLayer = 1
# pooledFeatMaps = tf.get_collection("pooledFeatureMaps")
# pollerPooledFeatMaps = cc.transformations.changeToPolar(pooledFeatureMaps[poolingLayer], pooledFeatureMaps[poolingLayer].get_shape()[-1].value)
# for a in angles:



# angles = [i*0.1 for i in range(6)]

numAngles = 7
layer=-1
image=1
featMapsMagns = tf.placeholder(tf.float32, [numAngles, actsMagn[layer].get_shape()[-3], actsMagn[layer].get_shape()[-2], actsMagn[layer].get_shape()[-1]])
featMapsAngs = tf.placeholder(tf.float32, [numAngles, actsMagn[layer].get_shape()[-3], actsMagn[layer].get_shape()[-2], actsMagn[layer].get_shape()[-1]])
featMapsCart = cc.transformations.changeToCartesian(featMapsMagns, featMapsAngs)
angles = [0, 15, 30, 45, 60, 75, 90]
for image in range(1000):
	data = numpy.zeros([len(angles), 28, 28, 1])
	for a in range(len(angles)):
		rotTest = numpy.fromfile("/scratch/data/rotateedTestSet/rotated_"+str(angles[a])+".npy")
		rotTest = numpy.reshape(rotTest, x_test.shape)
		data[a] = rotTest[image]
	angles2 = [[0], [15], [30], [45], [60], [75], [90]]
	inputPlaceholders = tf.get_collection("inputTestData")
	sess.run(testIterator.initializer, feed_dict={inputPlaceholders[0]: data, inputPlaceholders[1]: numpy.array(angles2)})
	am, aa = sess.run([actsMagn[layer], actsAngles[layer]])
	cart = sess.run(featMapsCart, feed_dict={featMapsMagns: am, featMapsAngs: aa})
	pickle.dump(cart, open("pkls/allccLCABNOR_"+str(image)+".pkl", "wb"))
	if image==0: print(cart.shape)


featMap = tf.placeholder(tf.float32, [1, actsMagn[layer].get_shape()[-3], actsMagn[layer].get_shape()[-2], 2*actsMagn[layer].get_shape()[-1]])
normVectorField = cc.layers.normalizeVectorField(featMap, actsMagn[layer].get_shape()[-3].value, actsMagn[layer].get_shape()[-2].value)
anTf = tf.placeholder(tf.float32, [])
manRot = cc.transformations.rotateVectorField(featMap, anTf, irelevantAxisFirst=True)
dists= numpy.zeros([len(angles),1000])
for a in range(len(angles)):
	for image in range(1000):
		cart = pickle.load(open("pkls/allccLCABNOR_"+str(image)+".pkl", "rb"))
		cartNorm = sess.run(normVectorField, feed_dict={featMap: cart[a:a+1]})
		manRotNp = sess.run(manRot, feed_dict={featMap: cart[0:1], anTf: angles[a]*numpy.pi/180})
		manRotNorm = sess.run(normVectorField, feed_dict={featMap: manRotNp})
		dists[a,image] = numpy.linalg.norm(manRotNorm-cartNorm)
	print("a = "+str(a)+": "+str(numpy.mean(dists[a,:])))





cols = 7
rows = 6
fig, axes = plt.subplots(nrows=rows, ncols=cols)
for i in range(cols):
	for j in range(rows):
		ax = fig.add_subplot(rows, cols, j*cols+i+1)
		ax.imshow(am[i,:,:,j])
		ax.set_xticks([])
		ax.set_yticks([])

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

cols = 7
rows = 6
fig, axes = plt.subplots(nrows=rows, ncols=cols)
for i in range(cols):
	for j in range(rows):
		axes[j,i].quiver(0, 0, cart[i,:,:,2*j+1], -cart[i,:,:,2*j], angles='xy', scale_units='xy', scale=1)
		axes[j,i].set_xlim(-1.5, 1.5)
		axes[j,i].set_ylim(-1.5, 1.5)
		axes[j,i].set_xticks([])
		axes[j,i].set_yticks([])
		# ax = fig.add_subplot(rows, cols, j*cols+i+1)
		# ax.imshow(am[i,:,:,j])

plt.show()


# am = []
# aa = []
# for a in range(len(angles)):
# aa = sess.run(actsAngles[layer])
# am = sess.run(actsMagn[layer], feed_dict={inptImage: data[a][:2]})
# aa = sess.run(actsAngles[layer], feed_dict={inptImage: data[a][:2]})

ra = []
featureMap = 1
# for a in angles:
# 	ra.append(sess.run(featMapRotPolar[layer], feed_dict={featMapsMagns[layer]: am[0][:,:,:,featureMap:featureMap+1], featMapsAngs[layer]: aa[0][:,:,:,featureMap:featureMap+1], rotationAngle: a}))

# cart = []
# for a in range(6):
	# cart.append(sess.run(featMapsCart[layer], feed_dict={featMapsMagns[layer]: am[a][:,:,:,featureMap:featureMap+1], featMapsAngs[layer]: aa[a][:,:,:,featureMap:featureMap+1]}))
# cart = sess.run(featMapsCart[layer], feed_dict={featMapsMagns[layer]: am[:,:,:,featureMap:featureMap+1], featMapsAngs[layer]: aa[:,:,:,featureMap:featureMap+1]})

# rotCart = sess.run(featMapRotations[layer], feed_dict={featMapsMagns[layer]: am[0][:,:,:,featureMap:featureMap+1], featMapsAngs[layer]: aa[0][:,:,:,featureMap:featureMap+1], rotationAngle: 0.2})

for a in range(6):
	plot_field(cart[a,0])
plot_field(rotCart[0])
plt.show()

rows = ["Layer 2", "Layer 5", "Layer 6"]
cols = ["0", "0.1 pi", "0.2 pi", "0.3 pi", "0.4 pi", "0.5 pi"]

shape = am[0].shape[-1]
fig, axes = plt.subplots(nrows=1, ncols=shape)
for i in range(shape):
	ax = fig.add_subplot(1, shape, i+1)
	ax.imshow(am[0][0,:,:,i])

shape = 6
fig, axes = plt.subplots(nrows=1, ncols=shape)
for i in range(shape):
	ax = fig.add_subplot(2, shape, i+1)
	ax.imshow(am[i][0,:,:,featureMap])
	ax = fig.add_subplot(2, shape, shape+i+1)
	ax.imshow(ra[i][0][0,:,:,0])

for ax, col in zip(axes[0], cols):
	ax.set_title(col)

for ax, row in zip(axes[:,0], rows):
	ax.set_ylabel(row, rotation=90, size='large')

ax = fig.add_subplot(2, 6, 1)
ax.imshow(am10[0,:,:,1])

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.show()
# ax.set_xticks([])
# ax.set_yticks([])
