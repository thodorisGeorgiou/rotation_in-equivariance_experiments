import numpy
import pickle
import tensorflow as tf
import cliffordConvolution as cc

mnistTrainGradientStd = 0.04170454
mnistTrainGradientStd = 0.041704543

def loadCifarData():
	cifar = tf.keras.datasets.cifar10
	(x_train, y_train),(x_test, y_test) = cifar.load_data()
	x_train, x_test = (x_train / 255.0).astype(numpy.float32), (x_test / 255.0).astype(numpy.float32)
	y_train, y_test = y_train.astype(numpy.int32), y_test.astype(numpy.int32)
	return [x_train, y_train], [x_test, y_test]


def loadMnistData():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = numpy.expand_dims((x_train / 255.0).astype(numpy.float32), -1), numpy.expand_dims((x_test / 255.0).astype(numpy.float32), -1)
	y_train, y_test = numpy.expand_dims(y_train.astype(numpy.int32), -1), numpy.expand_dims(y_test.astype(numpy.int32), -1)
	inds = pickle.load(open("mnistSmallInds.pkl","rb"))
	x_train, y_train = x_train[inds], y_train[inds]
	return [x_train[:40000], y_train[:40000]], [numpy.concatenate([x_test, x_train[40000:]], axis=0), numpy.concatenate([y_test, y_train[40000:]], axis=0)]


def constantAngle(batch_size):
	x_train = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_train = tf.placeholder(tf.int32, [None, 1])
	x_test = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_test = tf.placeholder(tf.int32, [None, 1])
	
	tf.add_to_collection("inputTrainData", x_train)
	tf.add_to_collection("inputTrainData", y_train)
	tf.add_to_collection("inputTestData", x_test)
	tf.add_to_collection("inputTestData", y_test)
	dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	dataset = dataset.shuffle(buffer_size=60000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10)
	# iterator = dataset.make_one_shot_iterator()
	iterator = dataset.make_initializable_iterator()

	next_example, next_label = iterator.get_next()
	next_example.set_shape([batch_size, 28, 28, 1])
	angs = tf.constant(numpy.ones([batch_size, 28, 28, 1],dtype=numpy.float32)*numpy.pi/4, dtype=tf.float32)
	next_example = cc.transformations.changeToCartesian(next_example, angs)
	next_example = cc.layers.normalizeVectorField(next_example, 3, 3)

	next_label.set_shape([batch_size, 1])
	next_label = tf.squeeze(next_label)

	testDataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	testDataset = testDataset.batch(batch_size)
	testIterator = testDataset.make_initializable_iterator()

	next_testExample, next_testLabel = testIterator.get_next()
	next_testExample.set_shape([batch_size, 28, 28, 1])
	next_testExample = cc.transformations.changeToCartesian(next_testExample, angs)
	next_testExample = cc.layers.normalizeVectorField(next_testExample, 3, 3)
	next_testLabel.set_shape([batch_size,1])
	next_testLabel = tf.squeeze(next_testLabel)
	return [next_example, next_label], [next_testExample, next_testLabel], [iterator, testIterator]

def rotateImage(image, label):
	ang = tf.random.uniform([], minval=-1, maxval=1)
	image = tf.contrib.image.rotate(image, ang*numpy.pi, interpolation="BILINEAR")
	return image, label

def gradientOrientation(batch_size, trainData=None, trainLabels=None, testBatch_size=None):
	if testBatch_size is None:
		testBatch_size=batch_size
	# [x_train, gx_train, y_train], [x_test, gx_test, y_test] = loadData()
	x_train = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_train = tf.placeholder(tf.int32, [None, 1])
	x_test = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_test = tf.placeholder(tf.int32, [None, 1])

	tf.add_to_collection("inputTrainData", x_train)
	tf.add_to_collection("inputTrainData", y_train)
	tf.add_to_collection("inputTestData", x_test)
	tf.add_to_collection("inputTestData", y_test)
	if numpy.any(trainData == None) or numpy.any(trainLabels == None):
		dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	else:
		dataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabels))
	dataset = dataset.map(rotateImage, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size=60000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size, drop_remainder=True)
	dataset = dataset.prefetch(100)
	iterator = dataset.make_one_shot_iterator()
	# iterator = dataset.make_initializable_iterator()
	blurKernel = tf.stop_gradient(cc.ops.getGuessKernel(1))

	next_example, next_label = iterator.get_next()
	next_example.set_shape([batch_size, 28, 28, 1])
	paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
	# next_example = tf.pad(next_example, paddings)

	# next_example = tf.nn.avg_pool(next_example, [1,3,3,1], [1,1,1,1], "SAME")
	# next_example = tf.nn.conv2d(next_example, blurKernel, [1,1,1,1], "SAME")
	next_example = cc.layers.calculateImageGradients(next_example)
	# next_example = next_example / mnistTrainGradientStd
	# next_example = cc.layers.normalizeVectorField(next_example, 3, 3)

	next_label.set_shape([batch_size,1])
	next_label = tf.squeeze(next_label)

	testDataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	testDataset = testDataset.batch(testBatch_size)
	testIterator = testDataset.make_initializable_iterator()
	next_testExample, next_testLabel = testIterator.get_next()
	next_testExample.set_shape([testBatch_size, 28, 28, 1])
	# next_testExample = tf.pad(next_testExample, paddings)

	# # paddings = tf.constant([[0,0], [2,2], [2,2], [0,0]])
	# next_testExample = tf.pad(next_testExample, paddings)


	# # next_testExample = tf.nn.avg_pool(next_testExample, [1,3,3,1], [1,1,1,1], "SAME")
	# # next_testExample = tf.nn.avg_pool(next_testExample, [1,3,3,1], [1,1,1,1], "SAME")
	# next_testExample = tf.nn.conv2d(next_testExample, blurKernel, [1,1,1,1], "SAME")
	next_testExample = cc.layers.calculateImageGradients(next_testExample)
	# next_testExample = next_testExample / mnistTrainGradientStd
	# next_testExample = cc.layers.normalizeVectorField(next_testExample, 3, 3)

	next_testLabel.set_shape([testBatch_size,1])
	next_testLabel = tf.squeeze(next_testLabel)
	return [next_example, next_label], [next_testExample, next_testLabel], [iterator, testIterator]

def gradientOrientationWithRotation(batch_size, fs=5):
	# [x_train, gx_train, y_train], [x_test, gx_test, y_test] = loadData()
	x_train = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_train = tf.placeholder(tf.int32, [None, 1])
	x_test = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_test = tf.placeholder(tf.int32, [None, 1])

	tf.add_to_collection("inputTrainData", x_train)
	tf.add_to_collection("inputTrainData", y_train)
	tf.add_to_collection("inputTestData", x_test)
	tf.add_to_collection("inputTestData", y_test)
	dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	dataset = dataset.shuffle(buffer_size=60000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10)
	# iterator = dataset.make_one_shot_iterator()
	iterator = dataset.make_initializable_iterator()

	next_example, next_label = iterator.get_next()
	next_example.set_shape([batch_size, 28, 28, 1])

	angle = tf.random.uniform([batch_size], minval=-1, maxval=1)
	next_example = tf.contrib.image.rotate(next_example, angle, interpolation="BILINEAR")

	next_example = cc.layers.calculateImageGradients(next_example)
	# next_example = tf.nn.avg_pool(next_example, [1,3,3,1], [1,1,1,1], "SAME")
	# angles = tf.unstack(angle)
	# next_examples = tf.split(next_example, batch_size)
	# next_examplel = []
	# for a, e in zip(angles, next_examples):
	# 	next_examplel.append(cc.transformations.rotateVectorField(e, a, irelevantAxisFirst=True))

	# next_example = cc.layers.normalizeVectorField(tf.concat(next_example, axis=0), fs, fs)
	# next_example = cc.layers.normalizeVectorField(next_example, fs, fs)

	next_label.set_shape([batch_size,1])
	next_label = tf.squeeze(next_label)

	testDataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	testDataset = testDataset.batch(batch_size)
	testIterator = testDataset.make_initializable_iterator()
	next_testExample, next_testLabel = testIterator.get_next()
	next_testExample.set_shape([batch_size, 28, 28, 1])
	next_testExample = cc.layers.calculateImageGradients(next_testExample)
	# next_testExample = tf.nn.avg_pool(next_testExample, [1,3,3,1], [1,1,1,1], "SAME")
	# next_testExample = cc.layers.normalizeVectorField(next_testExample, fs, fs)

	next_testLabel.set_shape([batch_size,1])
	next_testLabel = tf.squeeze(next_testLabel)
	return [next_example, next_label], [next_testExample, next_testLabel], [iterator, testIterator]

def colorWithRotation(batch_size):
	x_train = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_train = tf.placeholder(tf.int32, [None, 1])
	x_test = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_test = tf.placeholder(tf.int32, [None, 1])

	tf.add_to_collection("inputTrainData", x_train)
	tf.add_to_collection("inputTrainData", y_train)
	tf.add_to_collection("inputTestData", x_test)
	tf.add_to_collection("inputTestData", y_test)
	dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	dataset = dataset.shuffle(buffer_size=60000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10)
	# iterator = dataset.make_one_shot_iterator()
	iterator = dataset.make_initializable_iterator()

	next_example, next_label = iterator.get_next()
	next_example.set_shape([batch_size, 28, 28, 1])

	angle = tf.random.uniform([batch_size], minval=-1, maxval=1)
	next_example = tf.contrib.image.rotate(next_example, angle, interpolation="BILINEAR")

	next_label.set_shape([batch_size,1])
	next_label = tf.squeeze(next_label)

	testDataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	testDataset = testDataset.batch(batch_size)
	testIterator = testDataset.make_initializable_iterator()
	next_testExample, next_testLabel = testIterator.get_next()
	next_testExample.set_shape([batch_size, 28, 28, 1])

	next_testLabel.set_shape([batch_size,1])
	next_testLabel = tf.squeeze(next_testLabel)
	return [next_example, next_label], [next_testExample, next_testLabel], [iterator, testIterator]

def colorInput(batch_size, trainData=None, trainLabels=None, testBatch_size=None):
	if testBatch_size is None:
		testBatch_size=batch_size
	# [x_train, gx_train, y_train], [x_test, gx_test, y_test] = loadData()
	x_train = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_train = tf.placeholder(tf.int32, [None, 1])
	x_test = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y_test = tf.placeholder(tf.int32, [None, 1])

	tf.add_to_collection("inputTrainData", x_train)
	tf.add_to_collection("inputTrainData", y_train)
	tf.add_to_collection("inputTestData", x_test)
	tf.add_to_collection("inputTestData", y_test)
	if numpy.any(trainData == None) or numpy.any(trainLabels == None):
		dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	else:
		dataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabels))
	dataset = dataset.shuffle(buffer_size=60000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10)
	iterator = dataset.make_one_shot_iterator()
	# iterator = dataset.make_initializable_iterator()

	next_example, next_label = iterator.get_next()
	next_example.set_shape([batch_size, 28, 28, 1])

	next_label.set_shape([batch_size,1])
	next_label = tf.squeeze(next_label)

	testDataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	testDataset = testDataset.batch(testBatch_size)
	testIterator = testDataset.make_initializable_iterator()
	next_testExample, next_testLabel = testIterator.get_next()
	next_testExample.set_shape([testBatch_size, 28, 28, 1])

	next_testLabel.set_shape([testBatch_size,1])
	next_testLabel = tf.squeeze(next_testLabel)
	return [next_example, next_label], [next_testExample, next_testLabel], [iterator, testIterator]
