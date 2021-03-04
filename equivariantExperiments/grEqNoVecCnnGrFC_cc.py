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

def layer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, group=True, padding="SAME", first=True, useType="test", resuse_batch_norm=False, normalize="bn", wd=0.0001, steerable=True):
	# convW, convb = getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, wd=wd, mode="nl")
	num_bins = 16
	if group:
		weightMask = tf.ones([c_o, k_h, k_w, c_i*num_bins], tf.float32)
		weightMask = tf.contrib.image.rotate(weightMask, numpy.pi/4)
		weightMask = tf.transpose(weightMask, [1,2,3,0])
	else:
		weightMask = tf.ones([c_o, k_h, k_w, c_i], tf.float32)
		weightMask = tf.contrib.image.rotate(weightMask, numpy.pi/4)
		weightMask = tf.transpose(weightMask, [1,2,3,0])
	conv_in = cc.layers.convAllRotations(inpt, [k_h, k_w, c_i, c_o], k_h, s_h, s_w, first=first, padding=padding, num_bins=num_bins, weightMask=weightMask, steerable=steerable)
	conv_in = cc.layers.batch_norm(conv_in, useType, reuse=resuse_batch_norm)
	# conv_in = tf.transpose(conv_in, [0,1,2,4,3])
	convInShape = conv_in.get_shape().as_list()
	conv_in = tf.reshape(conv_in, convInShape[:-2]+[convInShape[-1]*convInShape[-2]])
	conv_relu = tf.nn.relu(conv_in)
	tf.add_to_collection("activationMagnitudes", conv_relu)
	return conv_relu

# def handCraftedFCLayer(inpt, weights, first=True, useType="test"):
# 	# angles = [a*2*numpy.pi/21 for a in range(21)]
# 	# nhfcW = numpy.array([[numpy.cos(an), numpy.sin(an)] for an in angles], dtype=numpy.float32)
# 	hfc = 
# 	return hfc

def ccLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=True, useType="test", resuse_batch_norm=False, normalize="vn", wd=0.0001):
	convW, convb = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, 2*c_i, s_h, s_w, wd=wd, lossType="nl")
	weightMask = tf.ones([c_o, k_h, k_w, 2*c_i], tf.float32)
	weightMask = tf.contrib.image.rotate(weightMask, numpy.pi/4)
	weightMask = tf.transpose(weightMask, [1,2,3,0])
	conv_in, angles = cc.layers.conv(inpt, convW, convb, c_i, c_o, s_h, s_w, first=first, useType=useType, padding=padding, num_bins=16, normalize=normalize, count=(useType=="train"), weightMask=weightMask)
	# conv_norm = tf.contrib.layers.group_norm(conv_in, 8, -1, (-3,-2), center=False, scale=False)
	tf.add_to_collection("activationMagnitudes", conv_in)
	tf.add_to_collection("activationAngles", angles)
	return conv_in, angles


def opLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=True, useType="test", resuse_batch_norm=False, normalize="bn", wd=0.0001, fc=False):
	convW, convb = getWeightsNBiases(first, k_h, k_w, c_o, 2*c_i, s_h, s_w, wd=wd, mode="nl")
	if fc:
		weightMask = None
	else:
		weightMask = tf.ones([c_o, k_h, k_w, 2*c_i], tf.float32)
		weightMask = tf.contrib.image.rotate(weightMask, numpy.pi/4)
		weightMask = tf.transpose(weightMask, [1,2,3,0])
	conv_in, angles = cc.layers.rotInvarianceWithArgMax(inpt, convW, convb, s_h, s_w, padding=padding, num_bins=32, weightMask=weightMask)
	if normalize == "bn":
		conv_in = cc.layers.batch_norm(conv_in, useType, reuse=resuse_batch_norm)
	elif normalize == False:
		pass
	else:
		exit("Not supported normalization mode.")
	conv_relu = tf.nn.relu(conv_in)
	tf.add_to_collection("activationMagnitudes", conv_relu)
	tf.add_to_collection("activationAngles", angles)
	# conv_in, angles = cc.layers.conv(inpt, convW, convb, c_i, c_o, s_h, s_w, padding=padding, normalize=normalize, count=False)
	# conv_norm = tf.contrib.layers.group_norm(conv_in, 8, -1, (-3,-2), center=False, scale=False)
	return conv_relu, angles

def getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, wd=0.0001, mode="wd"):
	if first:
		stddev=numpy.sqrt(2.0 / (c_i*k_h*k_w))
		if mode=="wd":
			convW = cc.misc._variable_with_weight_decay('weights', shape=[k_h, k_w, c_i, c_o], stddev=stddev, wd=wd)
		else:
			convW = cc.misc._variable_with_norm_loss('weights', shape=[k_h, k_w, c_i, c_o], stddev=stddev, nl=wd)

		convb = _variable('biases', [c_o], tf.constant_initializer(0.0))
		# if wd is not None:
		# 	weight_decay = tf.multiply(tf.nn.l2_loss(convW), wd, name='weight_loss')
		# 	tf.add_to_collection('losses', weight_decay)
	else:
		convW = tf.get_variable("weights")
		convb = tf.get_variable("biases")
	return convW, convb

def getFCWeightsNBiases(first, c_o, c_i, wd=0.0001):
	if first:
		# stddev=numpy.sqrt(2.0 / (c_i))
		convW = _variable_with_weight_decay('weights', shape=[c_i, c_o], stddev=0.01, wd=wd)
		convb = _variable('biases', [c_o], tf.constant_initializer(1.0))
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
	if useType=="train":
		wd = tf.Variable(1e-9, dtype=tf.float32, trainable=False)
	else:
		wd = 0.0001
	#conv1_1
	with tf.variable_scope("conv1_1", reuse=(not first)) as scope:
		k_h = 9; k_w = 9; c_i = 1; c_o = 24; s_h = 1; s_w = 1
		conv1_1_relu = layer(images, k_h, k_w, c_i, c_o, s_h, s_w, group=False, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd)
		conv1 = tf.image.crop_to_bounding_box(conv1_1_relu, 4, 4, 20, 20)
		tf.add_to_collection("cartesianRepresentationPooled", conv1)
	#conv2_1
	with tf.variable_scope("conv2_1", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_i = 24; c_o = 32; s_h = 1; s_w = 1
		conv2_1_relu = layer(conv1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd)
		conv2_1 = tf.nn.avg_pool(conv2_1_relu, [1,2,2,1], [1,2,2,1], padding='VALID')
	#conv2_2
	with tf.variable_scope("conv2_2", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_i = 32; c_o = 36; s_h = 1; s_w = 1
		conv2_2 = layer(conv2_1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd)
		tf.add_to_collection("cartesianRepresentationPooled", conv2_2)
	#conv3_1
	with tf.variable_scope("conv3_1", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_i = 36; c_o = 36; s_h = 1; s_w = 1
		conv3_1_relu = layer(conv2_2, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd)
		conv3_1 = tf.nn.avg_pool(conv3_1_relu, [1,2,2,1], [1,2,2,1], padding='VALID')
	#conv3_1.5
	with tf.variable_scope("conv3_1.5", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_i = 36; c_o = 64; s_h = 1; s_w = 1
		conv3_15 = layer(conv3_1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd)
		tf.add_to_collection("cartesianRepresentationPooled", conv3_15)
	conv3_2 = conv3_15
	shape = conv3_2.get_shape()
	with tf.variable_scope("conv4_1", reuse=(not first)) as scope:
		k_h = shape[-3].value; k_w = shape[-2].value; c_i = 64; c_o = 96; s_h = 1; s_w = 1
		conv4_1 = layer(conv3_2, k_h, k_w, c_i, c_o, s_h, s_w, padding="VALID", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd)
		# conv4_1_relu = tf.nn.avg_pool(conv4_1_relu, [1,k_h,k_w,1],[1,k_h,k_w,1], padding="VALID")
		# conv4_1 = tf.reduce_max(tf.reshape(conv4_1_relu, [batch_size,1,1,16,96]), axis=-2)
		tf.add_to_collection("activationMagnitudes", conv4_1)
		# if useType=="train":
		# 	conv4_1 = tf.nn.dropout(conv4_1, rate=0.3)
	#fc1
	with tf.variable_scope("fc1", reuse=(not first)) as scope:
		k_h = 1; k_w = 1; c_i = 96; c_o = 96; s_h = 1; s_w = 1
		fc1 = layer(conv4_1, k_h, k_w, c_i, c_o, s_h, s_w, padding="VALID", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd, steerable=False)
		tf.add_to_collection("activationMagnitudes", conv4_1)
		if useType=="train":
			fc1 = tf.nn.dropout(fc1, rate=0.2)
	#fc2
	with tf.variable_scope("fc2", reuse=(not first)) as scope:
		k_h = 1; k_w = 1; c_i = 96; c_o = 96; s_h = 1; s_w = 1
		fc2_relu = layer(fc1, k_h, k_w, c_i, c_o, s_h, s_w, padding="VALID", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=False, wd=wd, steerable=False)
		fc2_relu = tf.reshape(fc2_relu, [batch_size,1,1,16,96])
		fc2 = tf.reduce_max(fc2_relu, axis=-2)
		angles = [a*2*numpy.pi/16 for a in range(16)]
		angles_fc2 = tf.argmax(fc2_relu, axis=-2)
		angles_fc2 = tf.gather(angles, angles_fc2)
		tf.add_to_collection("activationMagnitudes", conv4_1)
		if useType=="train":
			fc2 = tf.nn.dropout(fc2, rate=0.2)
		fc2_cart = cc.transformations.changeToCartesian(fc2, angles_fc2)
	#out
	with tf.variable_scope("out", reuse=(not first)) as scope:
		k_h = 1; k_w = 1; c_i = 96; c_o = 10; s_h = 1; s_w = 1; num_bins = 16
		out, out_Angles = cc.layers.fcCliffordLayer(fc2_cart, c_i, c_o, s_h, s_w, first, useType, wd=0.0001, num_bins=num_bins, num_angles=4, normalize=False, resuse_batch_norm=resuse_batch_norm)
		prob = tf.nn.softmax(out, name="ClassProbabilities")

	return tf.squeeze(out), tf.squeeze(prob), out_Angles


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
