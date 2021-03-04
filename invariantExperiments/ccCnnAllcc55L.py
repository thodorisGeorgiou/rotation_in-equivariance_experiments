import os
import sys
import numpy

# sys.path.append("/tank/georgioutk/cliffordConvolutionNoReLU/")
import tensorflow as tf
import cliffordConvolution as cc

MOVING_AVERAGE_DECAY = 0.9999
bn_decay = 0.999
bn_epsilon = 1e-9

@tf.custom_gradient
def weightToThetaGrads(inpt, output, mask, weights, qThetas, thetas):
	def grad(dy):
		numAngels = weights.get_shape()[0].value
		w1Inds = tf.where(tf.equal(qThetas, 0), tf.ones_like(qThetas)*(numAngels-2), qThetas-1)
		w2Inds = tf.where(tf.equal(qThetas, numAngels-1), tf.ones_like(qThetas), qThetas+1)
		listWeights = tf.unstack(weights, axis=-1)
		lw1Inds = tf.unstack(w1Inds, axis=-1)
		lw2Inds = tf.unstack(w2Inds, axis=-1)
		w1 = []; w2 = []
		for i in range(qThetas.get_shape()[-1].value):
			w1.append(tf.gather(listWeights[i], lw1Inds[i]))
			w2.append(tf.gather(listWeights[i], lw2Inds[i]))
		w1 = tf.stack(w1, axis=3)
		w2 = tf.stack(w2, axis=3)
		normFactor = 4*numpy.pi/(numAngels-1)
		woa = (w2-w1)/normFactor
		exInpt = tf.expand_dims(inpt, axis=1)
		exInpt = tf.expand_dims(exInpt, axis=1)
		exInpt = tf.expand_dims(exInpt, axis=1)
		exInpt = tf.tile(exInpt, [1,1,1,output.get_shape()[-1].value,1,1,1])
		inptWoa = tf.reduce_sum(tf.multiply(exInpt, woa), axis=[-3,-2,-1])
		thetaGrads = tf.multiply(inptWoa, dy*mask)
		return (None, dy, None, None, None, thetaGrads)
	return tf.identity(output), grad

def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	return var

def _variable(name, shape, initializer, trainable=True):
	if trainable:
		var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	else:
		var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd, cpu=False):
	if cpu:
		var = _variable_on_cpu(name, shape, \
			tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	else:
		var = tf.get_variable(name, shape, \
			initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), dtype=tf.float32)
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def conv(inpt, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
	c_i = inpt.get_shape()[-1]
	assert c_i%group==0
	assert c_o%group==0
	convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
	if group==1:
		conv = convolve(inpt, kernel)
	else:
		input_groups = tf.split(inpt, num_or_size_splits=group, axis=3)
		kernel_groups = tf.split(kernel, num_or_size_splits=group, axis=3)
		output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		conv = tf.concat(output_groups, 3)
	return  tf.nn.bias_add(conv, biases)

def layer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=True, useType="test", resuse_batch_norm=False, normalize="vn", wd=0.0001):
	convW, convb = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, 2*c_i, s_h, s_w, wd=wd, lossType="nl")
	weightMask = tf.ones([c_o, k_h, k_w, 2*c_i], tf.float32)
	weightMask = tf.contrib.image.rotate(weightMask, numpy.pi/4)
	weightMask = tf.transpose(weightMask, [1,2,3,0])
	conv_in, angles = cc.layers.conv(inpt, convW, convb, c_i, c_o, s_h, s_w, first=first, useType=useType, padding=padding, num_bins=16, normalize=normalize, count=(useType=="train"), weightMask=weightMask)
	# conv_norm = tf.contrib.layers.group_norm(conv_in, 8, -1, (-3,-2), center=False, scale=False)
	tf.add_to_collection("activationMagnitudes", conv_in)
	tf.add_to_collection("activationAngles", angles)
	return conv_in, angles

# def getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, wd=0.0001, mode="wd"):
# 	if first:
# 		stddev=numpy.sqrt(2.0 / (c_i*k_h*k_w))
# 		if mode=="wd":
# 			convW = cc.misc._variable_with_weight_decay('weights', shape=[k_h, k_w, c_i, c_o], stddev=stddev, wd=wd)
# 		else:
# 			convW = cc.misc._variable_with_norm_loss('weights', shape=[k_h, k_w, c_i, c_o], stddev=stddev, nl=wd)

# 		convb = _variable('biases', [c_o], tf.constant_initializer(0.0))
# 		# if wd is not None:
# 		# 	weight_decay = tf.multiply(tf.nn.l2_loss(convW), wd, name='weight_loss')
# 		# 	tf.add_to_collection('losses', weight_decay)
# 	else:
# 		convW = tf.get_variable("weights")
# 		convb = tf.get_variable("biases")
# 	return convW, convb

def getFCWeightsNBiases(first, c_o, c_i, wd=0.0001):
	if first:
		# stddev=numpy.sqrt(2.0 / (c_i))
		convW = _variable_with_weight_decay('weights', shape=[c_i, c_o], stddev=0.01, wd=wd)
		convb = _variable('biases', [c_o], tf.constant_initializer(0.0))
		# if wd is not None:
		# 	weight_decay = tf.multiply(tf.nn.l2_loss(convW), wd, name='weight_loss')
		# 	tf.add_to_collection('losses', weight_decay)
	else:
		convW = tf.get_variable("weights")
		convb = tf.get_variable("biases")
	return convW, convb

def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg

def normalization_moving_averages():
	normalization_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
	normParams = tf.get_collection('normalizationParameters')
	norm_averages_op = normalization_averages.apply(normParams)
	return norm_averages_op, normalization_averages

def inference(images, batch_size, useType="test", first=True, resuse_batch_norm=False, fs=3, normalizationMode="bn"):
	count=False
	if useType=="train":
		wd = tf.Variable(1e-4, dtype=tf.float32, trainable=False)
	else:
		wd = 0.0001
	#conv1_1
	with tf.variable_scope("conv1_1", reuse=(not first)) as scope:
		k_h = 9; k_w = 9; c_i = 1; c_o = 16; s_h = 1; s_w = 1
		conv1_1_relu, angles_1_1 = layer(images, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=normalizationMode, wd=wd)
		conv1_1_relu = tf.image.crop_to_bounding_box(conv1_1_relu, 4, 4, 20, 20)
		angles_1_1 = tf.image.crop_to_bounding_box(angles_1_1, 4, 4, 20, 20)
		# conv1_1_relu = cc.layers.batch_norm_only_rescale_learn_scale(conv1_1_relu, useType, reuse=(not first))
		# conv1_1_relu = cc.layers.batch_norm_only_rescale(conv1_1_relu, useType, reuse=(not first))
		conv1 = cc.transformations.changeToCartesian(conv1_1_relu, angles_1_1, False)
		# conv1 = tf.nn.avg_pool(conv1, [1,2,2,1], [1,2,2,1], padding='VALID')
		# conv1 = cc.layers.batch_norm_only_rescale(conv1, useType, reuse=(not first))
		conv1 = cc.layers.normalizeVectorField(conv1, fs, fs)
		tf.add_to_collection("cartesianRepresentationPooled", conv1)
		if count:
			with tf.device('/cpu:0'):
				size = 1
				for shape_value in conv1.get_shape():
					size *= shape_value
				nonZero1_1 = tf.count_nonzero(conv1)
				tf.summary.scalar("nonZero1_1", nonZero1_1/size.value)
	#conv2_1
	with tf.variable_scope("conv2_1", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_i = 16; c_o = 32; s_h = 1; s_w = 1
		conv2_1_relu, angles_2_1 = layer(conv1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=normalizationMode, wd=wd)
		# conv2_1_relu = cc.layers.batch_norm_only_rescale_learn_scale(conv2_1_relu, useType, reuse=(not first))
		# conv2_1_relu = cc.layers.batch_norm_only_rescale(conv2_1_relu, useType, reuse=(not first))
		conv2_1 = cc.transformations.changeToCartesian(conv2_1_relu, angles_2_1, False)
		conv2_1 = tf.nn.avg_pool(conv2_1, [1,2,2,1], [1,2,2,1], padding='VALID')
		# conv2_1 = cc.layers.batch_norm_only_rescale(conv2_1, useType, reuse=(not first))
		conv2_1 = cc.layers.normalizeVectorField(conv2_1, fs, fs)
		# if useType=="train":
		# 	with tf.device('/cpu:0'):
		# 		size = 1
		# 		for shape_value in conv2_1.get_shape():
		# 			size *= shape_value
		# 		nonZero2_1 = tf.count_nonzero(conv2_1)
		# 		tf.summary.scalar("nonZero2_1", nonZero2_1/size.value)
	#conv2_2
	with tf.variable_scope("conv2_2", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_i = 32; c_o = 36; s_h = 1; s_w = 1
		conv2_2_relu, angles_2_2 = layer(conv2_1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=normalizationMode, wd=wd)
		# conv2_2_relu = cc.layers.batch_norm_only_rescale_learn_scale(conv2_2_relu, useType, reuse=(not first))
		# conv2_2_relu = cc.layers.batch_norm_only_rescale(conv2_2_relu, useType, reuse=(not first))
		conv2_2 = cc.transformations.changeToCartesian(conv2_2_relu, angles_2_2, False)
		# conv2_2 = tf.nn.avg_pool(conv2_2, [1,2,2,1], [1,2,2,1], padding='VALID')
		# conv2_2 = cc.layers.batch_norm_only_rescale(conv2_2, useType, reuse=(not first))
		conv2_2 = cc.layers.normalizeVectorField(conv2_2, fs, fs)
		tf.add_to_collection("cartesianRepresentationPooled", conv2_2)
		# maxpool2
		# with tf.variable_scope("maxPool2") as scope:
		# 	k_h = 2; k_w = 2; s_h = 2; s_w = 2; padding = 'VALID'
		# 	maxPool2, pool2Indexes = tf.nn.max_pool_with_argmax(conv2_2_relu, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="maxPool2")
		# 	intIndeces = tf.cast(pool2Indexes, tf.int32)
		# 	pooled_angles_2 = cc.ops.poolByIndex(angles_2_2, intIndeces)
		# 	conv2_2 = cc.layers.normalizeVectorField(maxPool2, fs, fs)
		# 	conv2_2 = cc.transformations.changeToCartesian(conv2_2, pooled_angles_2, False)
		# if useType=="train":
		# 	with tf.device('/cpu:0'):
		# 		size = 1
		# 		for shape_value in conv2_2.get_shape():
		# 			size *= shape_value
		# 		nonZero2_2 = tf.count_nonzero(conv2_2)
		# 		tf.summary.scalar("nonZero2_2", nonZero2_2/size.value)
	#conv3_1
	with tf.variable_scope("conv3_1", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_i = 36; c_o = 36; s_h = 1; s_w = 1
		conv3_1_relu, angles_3_1 = layer(conv2_2, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=normalizationMode, wd=wd)
		# conv3_1_relu = cc.layers.batch_norm_only_rescale_learn_scale(conv3_1_relu, useType, reuse=(not first))
		# conv3_1_relu = cc.layers.batch_norm_only_rescale(conv3_1_relu, useType, reuse=(not first))
		conv3_1 = cc.transformations.changeToCartesian(conv3_1_relu, angles_3_1, False)
		conv3_1 = tf.nn.avg_pool(conv3_1, [1,2,2,1], [1,2,2,1], padding='VALID')
		# conv3_1 = cc.layers.batch_norm_only_rescale(conv3_1, useType, reuse=(not first))
		conv3_1 = cc.layers.normalizeVectorField(conv3_1, fs, fs)
		# if useType=="train":
		# 	with tf.device('/cpu:0'):
		# 		size = 1
		# 		for shape_value in conv3_1.get_shape():
		# 			size *= shape_value
		# 		nonZero3_1 = tf.count_nonzero(conv3_1)
		# 		tf.summary.scalar("nonZero3_1", nonZero3_1/size.value)
	#conv3_1.5
	with tf.variable_scope("conv3_1.5", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_i = 36; c_o = 64; s_h = 1; s_w = 1
		conv3_15_relu, angles_3_15 = layer(conv3_1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=normalizationMode, wd=wd)
		# conv3_15_relu = cc.layers.batch_norm_only_rescale_learn_scale(conv3_15_relu, useType, reuse=(not first))
		# conv3_15_relu = cc.layers.batch_norm_only_rescale(conv3_15_relu, useType, reuse=(not first))
		conv3_15 = cc.transformations.changeToCartesian(conv3_15_relu, angles_3_15, False)
		# conv3_15 = tf.nn.avg_pool(conv3_15, [1,2,2,1], [1,2,2,1], padding='VALID')
		# conv3_15 = cc.layers.batch_norm_only_rescale(conv3_15, useType, reuse=(not first))
		conv3_15 = cc.layers.normalizeVectorField(conv3_15, 4, 4)
		tf.add_to_collection("cartesianRepresentationPooled", conv3_15)
	conv3_2 = conv3_15
	shape = conv3_2.get_shape()
	# #conv4_1
	# with tf.variable_scope("conv4_1", reuse=(not first)) as scope:
	# 	k_h = fs; k_w = fs; c_i = 64; c_o = 96; s_h = 1; s_w = 1
	# 	conv4_1_relu, angles_4_1 = layer(conv3_2, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd)
	# 	# conv4_1 = cc.transformations.changeToCartesian(conv4_1_relu, angles_4_1, False)
	# 	conv4_1 = conv4_1_relu
	# 	# if useType=="train":
	# 	# 	conv4_1 = tf.nn.dropout(conv4_1, rate=0.2)
	# 	conv4_1 = tf.nn.avg_pool(conv4_1, [1,4,4,1], [1,2,2,1], padding='VALID')
	# 	conv4_1 = cc.layers.batch_norm_only_rescale_learn_scale(conv4_1, useType, reuse=(not first))
	# shape = conv4_1.get_shape()
	with tf.variable_scope("conv4_1", reuse=(not first)) as scope:
		k_h = shape[-3].value; k_w = shape[-2].value; c_i = 64; c_o = 96; s_h = 1; s_w = 1
		print("Valid filter size = ", end="")
		print(k_h)
		conv4_1W, conv4_1b = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, 2*c_i, s_h, s_w, wd=wd, lossType="nl")
		weightMask = tf.ones([c_o, k_h, k_w, 2*c_i], tf.float32)
		weightMask = tf.contrib.image.rotate(weightMask, numpy.pi/4)
		weightMask = tf.transpose(weightMask, [1,2,3,0])
		# conv4_1W = conv4_1W*weightMask
		# if useType=="train" and first:
		# 	with tf.device('/cpu:0'):
				# tf.summary.histogram("angles_3_2", angles_3_2)
				# tf.summary.histogram("conv3_2_relu", conv3_2_relu)
				# tf.summary.histogram("conv3_2", conv3_2)

		num_bins = 16
		num_angles = 4
		offset = num_bins//(2*num_angles)
		# angles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
		angles = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
		try:
			multipliers = tf.constant(cc.misc.multipliers[num_bins][k_h])
			# multipliers = tf.constant([1.0 for i in range(num_bins+1)])
		except KeyError:
			exit("multipliers for filter size "+str(k_h)+" do not exist. Change filter size or add them manually.")
		rotatedWeightsL = []
		thetas = []
		conv0L = []
		activations0 = []
		activations2 = []
		for angle in angles:
			rotatedWeightsL.append(cc.transformations.rotateVectorField(conv4_1W, angle)*weightMask)
			# rotatedWeightsL.append(cc.transformations.rotateVectorField(conv4_1W, angle))
			# rw = cc.transformations.rotateVectorField(conv4_1W, angle, returnIrelevantAxisFirst=True)
			# rWeight = tf.image.resize(rw, tf.constant([int(k_h), int(k_h)], dtype=tf.int32))
			# rotatedWeightsL.append(tf.transpose(rWeight, [1,2,3,0])* weightMask)

		for angle in range(0, num_bins, num_bins//num_angles):
			weightSet = rotatedWeightsL[angle + offset]
			# scaledConv = conv3_2*25
			if useType == "train" and first:
				with tf.variable_scope(str(angle)) as scope:
					c42Conv2 = cc.layers.conv_2tf(conv3_2, weightSet, c_i, c_o, 1, 1, "VALID", monitor=False)
			else:
				c42Conv2 = cc.layers.conv_2tf(conv3_2, weightSet, c_i, c_o, 1, 1, "VALID")
			c42Conv0 = tf.nn.conv2d(conv3_2, weightSet, [1,1,1,1], "VALID")
			# if useType=="train" and first:
			# 	with tf.device('/cpu:0'):
			# 		tf.summary.histogram("conv2_"+str(angle), c42Conv2)
			# 		tf.summary.histogram("conv0_"+str(angle), c42Conv0)
			# angleMask = tf.cast(tf.logical_or(tf.abs(c42Conv0)>1e-2, tf.abs(c42Conv2)>1e-2), tf.float32)
			# angleMask = tf.abs(c42Conv2)<1e-3
			angleMask = (tf.square(c42Conv0) + tf.square(c42Conv2)) < 1.5e-4
			# angleMask = tf.logical_and(tf.abs(c42Conv0)<1e-2, tf.abs(c42Conv2)<1e-2)
			# if useType == "train" and first:
			# 	with tf.device('/cpu:0'):
			# 		size = 1
			# 		for shape_value in angleMask.get_shape():
			# 			size *= shape_value
			# 		nonZeroMask = tf.count_nonzero(tf.math.logical_not(angleMask))
			# 		tf.summary.scalar("nonZero_mask", nonZeroMask/size.value)
			c42Conv0 = tf.where(angleMask, tf.ones_like(c42Conv0)*(-1), c42Conv0)
			c42Conv2 = tf.where(angleMask, tf.zeros_like(c42Conv2), c42Conv2)
			# activations0.append(c42Conv0)
			# activations2.append(c42Conv2)
			# c42Conv2 = cc.layers.maskGradients(c42Conv2, angleMask)
			# c42Conv0 = cc.layers.maskGradients(c42Conv0, angleMask)
			thetas.append(tf.atan2(c42Conv2, c42Conv0))
			conv0L.append(c42Conv0)
			# if useType=="train":
			# 	with tf.device('/cpu:0'):
			# 		tf.summary.histogram("thetas", thetas[-1])

		thetas = tf.stack(thetas, axis=-1)
		conv0 = tf.stack(conv0L, axis=-1)
		# winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
		winner = tf.argmax(conv0, axis=-1, output_type=tf.int32)
		# convMask = tf.math.logical_not(angleMask)
		thetas2 = cc.ops.reduceIndex(thetas, winner)
		# if useType=="train" and first:
		# 	activations0 = tf.stack(activations0, axis=-1)
		# 	activations2 = tf.stack(activations2, axis=-1)
		# 	activations2Reduced = cc.ops.reduceIndex(activations2, winner)
		# 	activations0Reduced = cc.ops.reduceIndex(activations0, winner)
		# 	with tf.device('/cpu:0'):
		# 		tf.summary.histogram("thetas2", thetas2)
		# 		tf.summary.histogram("activations0", activations0Reduced)
		# 		tf.summary.histogram("activations2", activations2Reduced)
		thetas2, convMask = cc.ops.offsetCorrect(thetas2, [2*numpy.pi/num_angles])
		tf.add_to_collection("conv_mask_by_angle", convMask)
		# if useType=="train" and first:
		# 	with tf.device('/cpu:0'):
		# 		size = 1
		# 		for shape_value in thetas2.get_shape():
		# 			size *= shape_value
		# 		# nonZeroThetas = tf.count_nonzero(thetas2)
		# 		nonZeroMask = tf.count_nonzero(convMask)
		# 		# tf.summary.scalar("nonZero_thetas", nonZeroThetas/size.value)
		# 		tf.summary.scalar("nonZero_Convmask", nonZeroMask/size.value)

		winnerInd = tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
		quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + winnerInd
		# quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
		quantized = tf.where(quantized>num_bins , quantized-num_bins , quantized)
		quantized = tf.where(quantized<0, quantized+num_bins , quantized)
		floatMask = tf.cast(convMask, tf.float32)

		flatConvCudaResL = []
		for rotation in range(num_bins+1):
			rMask = tf.cast(tf.equal(quantized, rotation), tf.float32)
			flatConvCudaResL.append(tf.nn.conv2d(conv3_2, rotatedWeightsL[rotation], [1,1,1,1], "VALID")*rMask)
		rotatedWeights = tf.stack(rotatedWeightsL, axis=0)
		convRes = tf.reduce_sum(tf.stack(flatConvCudaResL, axis=0), axis=0)*floatMask
		convRes = weightToThetaGrads(conv3_2, convRes, floatMask, rotatedWeights, quantized, thetas2)
		angles = tf.concat(angles, axis=0)
		resAngleQuantized = tf.gather(angles, quantized)
		resAngle = thetas2 + tf.gather(angles, winnerInd)
		resMultiplier = tf.gather(multipliers, quantized)
		# resAngle = thetas2 + tf.cast(winnerInd,tf.float32)*2*numpy.pi/num_angles
		# resAngle2 = thetas2 + tf.cast(winner,tf.float32)*2*numpy.pi/num_angles

		diff = tf.abs(resAngle - resAngleQuantized)
		# convRes = convRes/tf.cos(diff)
		# convRes = convRes*resMultiplier/tf.cos(diff)
		convRes = convRes*resMultiplier/tf.stop_gradient(tf.cos(diff))
		# with tf.device('/cpu:0'):
		# 	convRes = cc.layers.monitorGrads(convRes)
		# conv4_1 = tf.nn.bias_add(convRes, conv4_1b)*floatMask
		# conv4_1 = batch_norm(conv4_1, useType, resuse_batch_norm) * floatMask
		# conv4_1 = cc.layers.batch_norm_only_rescale(conv4_1, useType, resuse_batch_norm) * floatMask
		conv4_1 = cc.layers.batch_norm(convRes, useType, resuse_batch_norm) * floatMask
		# conv4_1 = cc.layers.batch_norm_only_rescale_learn_scale(conv4_1, useType, resuse_batch_norm) * floatMask
		conv4_1 = tf.nn.relu(conv4_1)
		reluMask = tf.where(conv4_1>0, tf.ones_like(conv4_1), tf.zeros_like(conv4_1))
		regulatedAngles = cc.layers.maskAngleGradients(resAngle, reluMask)
		conv4_1_cart = cc.transformations.changeToCartesian(conv4_1, regulatedAngles, False)
		# conv4_1 = convRes
		# conv4_1 = tf.contrib.layers.group_norm(conv4_1, 1, -1, (-3,-2))
		# conv4_1 = tf.nn.leaky_relu(conv4_1, alpha=0.1)
		tf.add_to_collection("activationMagnitudes", conv4_1)
		tf.add_to_collection("cartesianRepresentationPooled", conv4_1_cart)
		tf.add_to_collection("activationAngles", resAngle)
		# tf.add_to_collection("activationAngles2", resAngle2)
		if useType=="train":
			conv4_1 = tf.nn.dropout(conv4_1, rate=0.2)
		# 	with tf.device('/cpu:0'):
		# 		perChannel = tf.unstack(conv4_1, axis=-1)
		# 		size = 1
		# 		for shape_value in conv4_1.get_shape()[:-1]:
		# 			size *= shape_value
		# 		for i in range(len(perChannel)):
		# 			nonZero4_1 = tf.count_nonzero(perChannel[i])
		# 			tf.summary.scalar("nonZero4_1_"+str(i), nonZero4_1/size.value)
	#fc1
	with tf.variable_scope("fc1", reuse=(not first)) as scope:
		fc1W, fc1b = getFCWeightsNBiases(first, 96, 96, wd=0.0001)
		fc1 = tf.matmul(tf.squeeze(conv4_1), fc1W) + fc1b
		fc1 = cc.layers.batch_norm(fc1, useType=useType, reuse=(not first))
		fc1 = tf.nn.leaky_relu(fc1, name=scope.name)
		if useType=="train":
			fc1 = tf.nn.dropout(fc1, rate=0.2)
	# #fc2
	with tf.variable_scope("fc2", reuse=(not first)) as scope:
		fc2W, fc2b = getFCWeightsNBiases(first, 96, 96, wd=0.0001)
		fc2 = tf.matmul(fc1, fc2W) + fc2b
		fc2 = cc.layers.batch_norm(fc2, useType=useType, reuse=(not first))
		fc2 = tf.nn.leaky_relu(fc2, name=scope.name)
		if useType=="train":
			fc2 = tf.nn.dropout(fc2, rate=0.2)

	#out
	with tf.variable_scope("out", reuse=(not first)) as scope:
		outW, outb = getFCWeightsNBiases(first, 10, 96, wd=0.0001)
		softmax_linear = tf.add(tf.matmul(fc2, outW), outb, name=scope.name)
		prob = tf.nn.softmax(softmax_linear, name="ClassProbabilities")
	return softmax_linear, prob, wd

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
		except tf.errors.OutOfRangeError:
			break
	return correct/count



def ema_to_weights(ema, variables):
	return tf.group(*(tf.assign(var, ema.average(var).read_value()) for var in variables))

def save_weight_backups():
	return tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(model_vars, backup_vars)))

def restore_weight_backups():
	return tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))

def to_training():
	with tf.control_dependencies([tf.assign(is_training, True)]):
		return restore_weight_backups()

def to_testing(ema):
	with tf.control_dependencies([tf.assign(is_training, False)]):
		with tf.control_dependencies([save_weight_backups()]):
			return ema_to_weights(ema, model_vars)


if __name__ == '__main__':
	if tf.gfile.Exists(train_dir):
		tf.gfile.DeleteRecursively(train_dir)
	tf.gfile.MakeDirs(train_dir)

	is_training = tf.get_variable('is_training', shape=(), dtype=tf.bool, initializer=tf.constant_initializer(True, dtype=tf.bool), trainable=False)

	global_step = tf.Variable(0, trainable=False)

	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = numpy.expand_dims((x_train / 255.0).astype(numpy.float32), -1), numpy.expand_dims((x_test / 255.0).astype(numpy.float32), -1)
	y_train, y_test = y_train.astype(numpy.int32), y_test.astype(numpy.int32)

	dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	dataset = dataset.shuffle(buffer_size=x_train.shape[0])
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10)
	iterator = dataset.make_one_shot_iterator()

	next_example, next_label = iterator.get_next()
	next_example.set_shape([batch_size, 28, 28, 1])
	next_label.set_shape([batch_size])

	testDataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	testDataset = testDataset.batch(batch_size)
	testIterator = testDataset.make_initializable_iterator()
	next_testExample, next_testLabel = testIterator.get_next()
	next_testExample.set_shape([batch_size, 28, 28, 1])
	next_testLabel.set_shape([batch_size])


	softmax_linear, prob, weightDecayFactor = inference(next_example, "train", first=True, resuse_batch_norm=False)

	testSoftmax, testProb, skoupidia = inference(next_testExample, "test", first=False, resuse_batch_norm=True)
	top_1 = tf.nn.in_top_k(testProb, next_testLabel, 1)

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_linear, labels=next_label, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

	loss_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	num_batches_per_epoch = x_train.shape[0] / batch_size
	decay_steps = int(num_batches_per_epoch * 2)

	lr = tf.Variable(1e-3, dtype=tf.float32, trainable=False)
	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		# opt = tf.train.GradientDescentOptimizer(lr)
		opt = tf.train.AdamOptimizer(lr)
		grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Track the moving averages of all trainable variables.
	model_vars = tf.trainable_variables()
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(model_vars)

	for l in tf.get_collection("losses") + [total_loss]:
		tf.summary.scalar(l.op.name +' (raw)', l)

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	with tf.variable_scope('BackupVariables'):
		backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False, initializer=var.initialized_value()) for var in model_vars]

	empty_op = lambda: tf.group()
	to_test_op = to_testing(variable_averages)
	to_train_op = to_training()

	saver = tf.train.Saver(tf.all_variables())
	init = tf.global_variables_initializer()
	myconfig = tf.ConfigProto(log_device_placement=False)
	myconfig.gpu_options.allow_growth = True
	sess = tf.Session(config=myconfig)
	writer = tf.summary.FileWriter(train_dir, sess.graph)
	sess.run(init)
	log = open("7LccCNN.txt", "w", 1)
	_summ = tf.summary.merge_all()

	SuccRateBefore = None
	SuccRateAfter = None
	SuccRate_summary = tf.Summary()
	SuccRate_summary.value.add(tag='raw_test_error', simple_value=SuccRateBefore)
	SuccRate_summary.value.add(tag='test_error', simple_value=SuccRateAfter)

	# for step in range(int(20*x_train.shape[0])):
	for step in range(int(40*x_train.shape[0]/batch_size)):
		exEntropy, totalLoss, summ, _ = sess.run([cross_entropy_mean, total_loss, _summ, train_op])
		writer.add_summary(summ, step)
		print(str(step)+" "+str(exEntropy), file=log)
		if step % (x_train.shape[0]//batch_size) == 0:
			print("Saving..")
			checkpoint_path = os.path.join(train_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=step)
		if step % (x_train.shape[0]//(batch_size*16)) == 0 and step != 0:
			SuccRateBefore = testNetwork(sess, top_1, batch_size, testIterator)
			print("Before switch:"+str(SuccRateBefore), file=log)
			sess.run(to_test_op)
			SuccRateAfter = testNetwork(sess, top_1, batch_size, testIterator)
			print("After switch :"+str(SuccRateAfter), file=log)
			SuccRate_summary.value[0].simple_value = SuccRateBefore
			SuccRate_summary.value[1].simple_value = SuccRateAfter
			writer.add_summary(SuccRate_summary, step)
			sess.run(to_train_op)

	print("Saving..")
	checkpoint_path = os.path.join(train_dir, 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=step)
