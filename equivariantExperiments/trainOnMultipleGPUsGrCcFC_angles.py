import os
import sys
import numpy
import pickle
import scipy.ndimage
# from multiprocessing import Pool
import multiprocessing
sys.path.append("/scratch/georgioutk/cliffordConvolution/")
import tensorflow as tf
import cliffordConvolution as cc
import preprocessing

# import ccCnnCcFC as network
# import grEqNoVecCnnGrFC_cc as network
import grEqNoVecCnnGrFC_sumCc as network
# import perCInCcCnnCcFC as network
# import opCnnAllcc55L as network
numGpus = 1
numGpusTest = 1
numCpus = 5
numEpochs = 360
batch_size = 128
testBatch_size = 400
MOVING_AVERAGE_DECAY = 0.0
run = sys.argv[1]
baseDir = "trained_complexWeights/grCcFCNet5L3x96_64_2x36_32_16_fs_9_all_7_16Bins_weightMask_trainAll_onlyRescaleBNFC_dropout_3x0.5_0MAD_droppingLR_30_"+run+"_"
# baseDir = "trained_testGrads/grCcFCNet5L64_2x96_64_2x36_32_16_fs_9_all_7_16Bins_weightMask_trainAll_onlyRescaleBNFC_dropout_3x0.5_0MAD_droppingLR_30_"+run+"_"
train_dir = os.getcwd()+"/"+baseDir

def rotateDataset(dataset=None, output=None):
	res = numpy.zeros(dataset.shape)
	for i in range(dataset.shape[0]):
		a = numpy.random.rand()*2*numpy.pi
		res[i] = scipy.ndimage.rotate(dataset[i], numpy.degrees(a), order=3, reshape=False)
	# return res
	output['x'] = res

def testNetwork(sess, top_1, testBatch_size, iterator, gx_test, y_test, a_test, log):
	inputPlaceholders = tf.get_collection("inputTestData")
	sess.run(iterator.initializer, feed_dict={inputPlaceholders[0]: gx_test, inputPlaceholders[1]: y_test, inputPlaceholders[2]: a_test})
	correct = 0
	xEntropy = 0
	anDif = 0
	count = 0
	while True:
		try:
			res = sess.run(top_1)
			correct += numpy.sum(res[0])
			xEntropy += res[1]
			anDif += res[2]
			count += testBatch_size
		except tf.errors.OutOfRangeError:
			break
	# print("Validation accuracy: "+str(correct/count), file=log)
	# return correct/count
	return 2.3 - xEntropy/count, correct/count, anDif/count

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

	# mnsitTrain = numpy.load("/data/georgioutk/experiments/mnistRot/data/mnist_rotation_new/rotated_train.npz")
	# mnsitVal = numpy.load("/data/georgioutk/experiments/mnistRot/data/mnist_rotation_new/rotated_valid.npz")
	# mnsitTrainVal = numpy.loadtxt("/data/georgioutk/experiments/mnistRot/data/mnist_rotation/mnist_all_rotation_normalized_float_train_valid.amat", dtype=numpy.float32)

	# mnsitTrainVal = numpy.loadtxt("data/mnist_all_rotation_normalized_float_train_valid.amat", dtype=numpy.float32)
	# mnsitTestRaw = numpy.loadtxt("data/mnist_all_rotation_normalized_float_test.amat", dtype=numpy.float32)
	# mnsitTrain = {}
	# mnsitVal = {}
	# mnsitTest = {}
	# mnsitTrainFull = {}
	# mnsitTrain['x'] = mnsitTrainVal[:,:-1]
	# mnsitTrain['y'] = mnsitTrainVal[:,-1].astype(numpy.int32)
	# mnsitTrainFull['x'] = mnsitTrainVal[:,:-1]
	# mnsitTrainFull['y'] = mnsitTrainVal[:,-1].astype(numpy.int32)
	# mnsitTest['x'] = mnsitTestRaw[:,:-1]
	# mnsitTest['y'] = mnsitTestRaw[:,-1].astype(numpy.int32)

	mnsitTrain = {'x': numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_imanges.npy"), \
	'y': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_labels.npy")[:,:1]).astype(numpy.int32), \
	'a': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_train_labels.npy")[:,1:2])}
	mnsitTest = {'x': numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_imanges.npy"), \
	'y': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_labels.npy")[:,:1]).astype(numpy.int32), \
	'a': (numpy.load("/scratch/georgioutk/mnist/my_mnist_rot/mnist_rot_test_labels.npy")[:,1:2])}
	# manager = multiprocessing.Manager()
	# trainSet = []
	# threads = []
	# for i in range(numCpus):
	# 	trainSet.append(manager.dict())
	# 	p = multiprocessing.Process(target=rotateDataset, args=(), kwargs={'dataset':mnsitTrain['x'].reshape([10000,28,28,1]), 'output':trainSet[-1]})
	# 	threads.append(p)
	# 	p.start()
	currLr = 1e-3
	lr = tf.Variable(currLr, dtype=tf.float32, trainable=False)
	tf.add_to_collection("learning_rate", lr)

	# [next_example, next_label], [next_testExample, next_testLabel], [trainIterator, testIterator] = preprocessing.colorInput(batch_size, mnsitTrain['x'].reshape([10000,28,28,1]), mnsitTrain['y'].reshape([10000,1]))
	[next_example, next_label, next_angle], [next_testExample, next_testLabel, next_testAngle], [trainIterator, testIterator] = \
	preprocessing.colorWithRotation(batch_size, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'], testBatch_size=testBatch_size)
	# [next_example, next_label], [next_testExample, next_testLabel], [trainIterator, testIterator] = preprocessing.gradientOrientation(batch_size, mnsitTrain['x'].reshape([12000,28,28,1]), mnsitTrain['y'].reshape([12000,1]), testBatch_size=testBatch_size)
	# [next_example, next_label], [next_testExample, next_testLabel], [trainIterator, testIterator] = preprocessing.gradientOrientation(batch_size, testBatch_size=testBatch_size)
	# paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
	# next_example = tf.pad(next_example, paddings)
	# next_testExample = tf.pad(next_testExample, paddings)
	nex = tf.split(next_example, numGpus, axis=0)
	nla = tf.split(next_label, numGpus, axis=0)
	nan = tf.split(next_angle, numGpus, axis=0)

	test_nex = tf.split(next_testExample, numGpusTest, axis=0)
	test_nla = tf.split(next_testLabel, numGpusTest, axis=0)

	for i in range(numGpus):
		with tf.name_scope('tower_%d' % (i)) as scope:
			with tf.device('/gpu:%d' % i):
				print("Defining tower "+str(i))
				softmax_linear, prob, pred_angle = network.inference(nex[i], batch_size//numGpus, "train", first=(i==0), resuse_batch_norm=False, fs=7, normalizationMode="bn")
				pred_angle = tf.squeeze(cc.ops.reduceIndex(pred_angle, tf.expand_dims(tf.expand_dims(nla[i], axis=-1), axis=-1)))
				# with tf.device('/cpu:0'):
				# 	softmax_linear = cc.layers.monitorGrads(softmax_linear)
				# 	pred_angle = cc.layers.monitorGrads(pred_angle)
				cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_linear, labels=nla[i], name='cross_entropy_per_example_'+str(i))
				# nan2 = nan[i] - numpy.pi
				angleDif = nan[i] - pred_angle
				# angleDif2 = nan2 - pred_angle
				normAngleDif = tf.abs(tf.atan2(tf.sin(angleDif), tf.cos(angleDif)))
				# normAngleDif2 = tf.abs(tf.atan2(tf.sin(angleDif2), tf.cos(angleDif2)))
				# normAngleDif = tf.reduce_min(tf.stack([normAngleDif, normAngleDif2], axis=0), axis=0)
				meanAnDif = tf.reduce_mean(normAngleDif, name="angle_loss_"+str(i))
				cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_'+str(i))
				tf.add_to_collection('x_entropies', cross_entropy_mean)
				tf.add_to_collection('angle_losses', meanAnDif)

	# softmax_linear, prob, weightDecayFactor = inference(next_example, batch_size, "train", first=True, resuse_batch_norm=False)
	# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_linear, labels=next_label, name='cross_entropy_per_example')
	# cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	# tf.add_to_collection('losses', cross_entropy_mean)

	tf.add_to_collection('losses', tf.reduce_mean(tf.get_collection('x_entropies')))
	tf.add_to_collection('losses', tf.reduce_mean(tf.get_collection('angle_losses')))
	print("All towers defined.")

	netOut = []
	netLogits = []
	netAngles = []
	for i in range(numGpusTest):
		with tf.name_scope('tower_%d' % (i)) as scope:
			with tf.device('/gpu:%d' % i):
				testSoftmax, testProb, testPredAngle = network.inference(test_nex[i], testBatch_size//numGpusTest, "test", first=False, resuse_batch_norm=True, fs=7, normalizationMode="bn")
				testPredAngle = tf.squeeze(cc.ops.reduceIndex(testPredAngle, tf.expand_dims(tf.expand_dims(test_nla[i], axis=-1), axis=-1)))
				netOut.append(testProb)
				netLogits.append(testSoftmax)
				netAngles.append(testPredAngle)

	netLogits = tf.concat(netLogits, axis=0)
	netAngles = tf.concat(netAngles, axis=0)
	# next_testAngle2 = tf.where(tf.logical_or(tf.equal(next_testLabel, 0), tf.equal(next_testLabel, 8)), next_testAngle-numpy.pi, next_testAngle)
	anDif = next_testAngle - netAngles
	# anDif2 = next_testAngle2 - netAngles
	testAnDif = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(anDif), tf.cos(anDif))))
	# testAnDif2 = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(anDif2), tf.cos(anDif2))))
	# testAnDif = tf.reduce_min(tf.stack([testAnDif, testAnDif2], axis=0), axis=0)
	testXEntropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=netLogits, labels=next_testLabel))
	top_1 = tf.nn.in_top_k(tf.concat(netOut, axis=0), next_testLabel, 1)
	print("Test towers defined.")
	# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_linear, labels=next_label, name='cross_entropy_per_example')
	# cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	# tf.add_to_collection('losses', cross_entropy_mean)

	total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

	loss_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		# opt = tf.train.GradientDescentOptimizer(lr)
		opt = tf.train.AdamOptimizer(lr)
		# opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
		# grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())
		grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables(), colocate_gradients_with_ops=True)
		# for i, g in enumerate(grads):
		# 	try:
		# 		print(str(i)+": "+g[1].name)
		# 		norm = tf.norm(g[0], name="Norm_"+str(i))
		# 		tf.summary.scalar(norm.op.name +' (raw)', norm)
		# 	except ValueError:
		# 		print(str(i)+" is NonType")
		# 		print(g[1].get_shape())
		# exit()
			# print("---------")
			# print(len(g))
			# print(type(g[0]))
			# print(type(g[1]))
			# print(g[0].get_shape())
			# print(g[1].get_shape())
		# 	norm = tf.norm(g[0])
		# print(len(grads))
		# print(grads[0][0].get_shape())
		# print(grads[1][0].get_shape())
		# print(grads[0][1].get_shape())
		# print(grads[1][1].get_shape())
		# exit()

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	regOps = tf.get_collection("regularizationOps")
	# Track the moving averages of all trainable variables.
	model_vars = tf.trainable_variables()
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(model_vars)

	for l in tf.get_collection("losses") + [total_loss]:
		tf.summary.scalar(l.op.name +' (raw)', l)

	for l in tf.get_collection("x_entropies"):
		tf.summary.scalar(l.op.name +' (raw)', l)

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	with tf.variable_scope('BackupVariables'):
		backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False, initializer=var.initialized_value()) for var in model_vars]

	empty_op = lambda: tf.group()
	to_test_op = to_testing(variable_averages)
	to_train_op = to_training()

	saver = tf.train.Saver(tf.global_variables())
	saverMax = tf.train.Saver(tf.global_variables())
	# idcs = pickle.load(open("mnistRandInds.pkl", "rb"))

	inputPlaceholders = tf.get_collection("inputTrainData")
	init = tf.global_variables_initializer()
	myconfig = tf.ConfigProto(log_device_placement=False)
	myconfig.gpu_options.allow_growth = True
	sess = tf.Session(config=myconfig)
	writer = tf.summary.FileWriter(train_dir, sess.graph)
	writerMax = tf.summary.FileWriter(train_dir[:-1]+"Release/", sess.graph)
	sess.run(init)
	# print("Stop preparing data")
	# x_train = []
	# y_train = []
	# for t in range(numCpus):
	# 	threads[t].join()
	# 	x_train.append(trainSet[t]["x"])
	# 	y_train.append(mnsitTrain['y'].reshape([10000,1]))
	# 	trainSet[t]["x"] = None
	# 	trainSet[t]["y"] = None

	# x_train = numpy.concatenate(x_train, axis=0)
	# y_train = numpy.concatenate(y_train, axis=0)

	# print("Start preparing data")
	# threads = []
	# for i in range(numCpus):
	# 	p = multiprocessing.Process(target=rotateDataset, args=(), kwargs={'dataset':mnsitTrain['x'].reshape([10000,28,28,1]), 'output':trainSet[i]})
	# 	threads.append(p)
	# 	threads[i].start()

	# sess.run(trainIterator.initializer, feed_dict={inputPlaceholders[0]: x_train, inputPlaceholders[1]: y_train})
	log = open(baseDir+".txt", "w", 1)
	# log = open("cc6LHalfWidth256LastVectorNormTraininAugmentationNewRotationPadBeforeGradsNL360Epochs.txt", "w", 1)
	_summ = tf.summary.merge_all()

	max_val = 0
	max_test = 0
	SuccRateValidation = None
	SuccRateTest = None
	SuccRate_summary = tf.Summary()
	SuccRate_summary.value.add(tag='validation_accuracy', simple_value=SuccRateValidation)
	SuccRate_summary.value.add(tag='max_validation_accuracy', simple_value=max_val)
	stepsPerEpoch = mnsitTrain['x'].shape[0]//batch_size
	scores = []
	for step in range(int(numEpochs*mnsitTrain['x'].shape[0]/batch_size)):
		# if step % (mnsitTrain['x'].shape[0]*15//(batch_size*0.25)) == 0 and (currLr > 1e-4) and step > 0:
		# 	currLr = 1e-4
		# 	print(str(step)+": learning rate = "+str(currLr))
		# 	lr.load(currLr, sess)
		if step % (mnsitTrain['x'].shape[0]*30//(batch_size*0.25)) == 0 and (currLr > 1e-5) and step > 0:
			currLr = 1e-5
			print(str(step)+": learning rate = "+str(currLr))
			lr.load(currLr, sess)
		# if step % (mnsitTrain['x'].shape[0]*60//(batch_size*0.25)) == 0 and (currLr == 1e-5) and step > 0:
		# 	currLr = 2e-6
		# 	print(str(step)+": learning rate = "+str(currLr))
		# 	lr.load(currLr, sess)
		# if step % (mnsitTrain['x'].shape[0]*60//(batch_size*0.25)) == 0 and (currLr > 1e-6) and step > 0:
		# 	currLr = 1e-7
		# 	print(str(step)+": learning rate = "+str(currLr))
		# 	lr.load(currLr, sess)
		# if step % (mnsitTrain['x'].shape[0]*70//(batch_size*0.25)) == 0 and (currLr > 1e-7) and step > 0:`
		# 	currLr = 1e-7
		# 	print(str(step)+": learning rate = "+str(currLr))
		# 	lr.load(currLr, sess)
		# if step >= 15*stepsPerEpoch and (step % stepsPerEpoch) == 0:
		# 	currLr *= 0.8
		# 	print("learning rate = "+str(currLr), file=log)
		# 	lr.load(currLr, sess)
		# if step == 10000:
		# 	currLr *= 0.8
		# 	print("learning rate = "+str(currLr), file=log)
		# 	lr.load(currLr, sess)
		# if step == 20000:
		# 	currLr *= 0.8
		# 	print("learning rate = "+str(currLr), file=log)
		# 	lr.load(currLr, sess)
		# if step == 22000:
		# 	currLr *= 0.2
		# 	print("learning rate = "+str(currLr), file=log)
		# 	lr.load(currLr, sess)
		__ = sess.run(regOps)
		exEntropy, totalLoss, summ, _ = sess.run([cross_entropy_mean, total_loss, _summ, train_op])
		# try:
		# 	exEntropy, totalLoss, summ, _ = sess.run([cross_entropy_mean, total_loss, _summ, train_op])
		# except tf.errors.OutOfRangeError:
		# 	x_train = []
		# 	y_train = []
		# 	for t in range(numCpus):
		# 		threads[t].join()
		# 		x_train.append(trainSet[t]["x"])
		# 		y_train.append(mnsitTrain['y'].reshape([10000,1]))
		# 		trainSet[t]["x"] = None
		# 		trainSet[t]["y"] = None
		# 	x_train = numpy.concatenate(x_train, axis=0)
		# 	y_train = numpy.concatenate(y_train, axis=0)
		# 	sess.run(trainIterator.initializer, feed_dict={inputPlaceholders[0]: x_train, inputPlaceholders[1]: y_train})
		# 	threads = []
		# 	for i in range(numCpus):
		# 		p = multiprocessing.Process(target=rotateDataset, args=(), kwargs={'dataset':mnsitTrain['x'].reshape([10000,28,28,1]), 'output':trainSet[i]})
		# 		threads.append(p)
		# 		threads[i].start()
		# 	continue
		writer.add_summary(summ, step)
		# print(str(step)+" "+str(exEntropy), file=log)
		if step % (mnsitTrain['x'].shape[0]//(batch_size*0.25)) == 0:
			print("%2.2f"%(step*100/(int(numEpochs*mnsitTrain['x'].shape[0]/batch_size))), end="\r", flush=True)
			checkpoint_path = os.path.join(train_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=step)
		if step % (mnsitTrain['x'].shape[0]//(batch_size*0.25)) == 0 and step != 0:
			sess.run(to_test_op)
			# valExEntropy, SuccRateValidation = testNetwork(sess, [top_1, testXEntropy], testBatch_size, testIterator, mnsitVal['x'].reshape([2000,28,28,1]), mnsitVal['y'].reshape([2000,1]), log)
			valExEntropy, SuccRateValidation, anDifValidation = testNetwork(sess, [top_1, testXEntropy, testAnDif], testBatch_size, testIterator, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'], log)
			# trainFullExEntropy, SuccRateTrainFull = testNetwork(sess, [top_1, testXEntropy], testBatch_size, testIterator, mnsitTrainFull['x'].reshape([12000,28,28,1]), mnsitTrainFull['y'].reshape([12000,1]), log)
			TestExEntropy, SuccRateTest, anDifTest = testNetwork(sess, [top_1, testXEntropy, testAnDif], testBatch_size, testIterator, mnsitTest['x'], mnsitTest['y'], mnsitTest['a'], log)
			# print("Validation: "+str(SuccRateValidation)+"/"+str(max_val), file=log)
			print("Scores: "+str([valExEntropy, TestExEntropy, SuccRateValidation, SuccRateTest, anDifValidation, anDifTest]), file=log)
			scores.append([valExEntropy, SuccRateValidation, TestExEntropy, SuccRateTest])
			if valExEntropy > max_val:
				max_val = valExEntropy
				checkpoint_path = os.path.join(train_dir[:-1]+"Release/", 'model.ckpt')
				saverMax.save(sess, checkpoint_path, global_step=step)
				writerMax.add_summary(summ, step)
			SuccRate_summary.value[0].simple_value = valExEntropy
			SuccRate_summary.value[1].simple_value = max_val
			writer.add_summary(SuccRate_summary, step)
			sess.run(to_train_op)
		# if step % 30*mnsitTrain['x'].shape[0]//batch_size == 0 and step != 0:
		# 	currLr /= 10
		# 	lr.load(currLr, sess)

	print("Saving..")
	pickle.dump(scores, open("scores.pkl", "wb"))
	checkpoint_path = os.path.join(train_dir, 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=step)
	sess.run(to_test_op)
	valExEntropy, SuccRateValidation, anDifValidation = testNetwork(sess, [top_1, testXEntropy, testAnDif], testBatch_size, testIterator, mnsitTrain['x'], mnsitTrain['y'], mnsitTrain['a'], log)
	# valExEntropy, SuccRateValidation = testNetwork(sess, [top_1, testXEntropy], testBatch_size, testIterator, mnsitVal['x'].reshape([2000,28,28,1]), mnsitVal['y'].reshape([2000,1]), log)
	print("Validation: "+str(SuccRateValidation)+"/"+str(max_val)+" - "+str(anDifValidation), file=log)
	SuccRate_summary.value[0].simple_value = valExEntropy
	if valExEntropy > max_val:
		max_val = valExEntropy
		checkpoint_path = os.path.join(train_dir[:-1]+"Release/", 'model.ckpt')
		saverMax.save(sess, checkpoint_path, global_step=step)
		writerMax.add_summary(summ, step)
	SuccRate_summary.value[1].simple_value = max_val
	writer.add_summary(SuccRate_summary, step)
	sess.run(to_train_op)
