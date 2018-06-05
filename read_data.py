import matplotlib.pyplot as plot

data = open('nohup.out')

line = data.readline()
while line:
	dev_accs = []
	dev_iters = []
	iters = 0
	eta, nodes = data.readline().split(' ')
	line = data.readline()
	while line.split(':')[0] == 'total datapoints':
		true_pos = data.readline()
		true_neg = data.readline()
		false_pos = data.readline()
		false_neg = data.readline()
		dev_acc = data.readline()
		avg_loss = data.readline()

		dev_iters.append(iters)
		iters += 200
		dev_accs.append(dev_acc)

		data.readline()
		line = data.readline()
	data.readline()
	test_true_pos = data.readline()
	test_true_neg = data.readline()
	test_false_pos = data.readline()
	test_false_neg = data.readline()
	test_acc = data.readline()
	print test_acc
	test_avg_loss = data.readline()
	data.readline()
	line = data.readline()
	plot.style.use('ggplot')
	plot.plot(dev_iters, dev_accs)
	plot.xlabel('Number of iterations')
	plot.ylabel('Validation Prediction Accuracy')
	plot.title('Dev Set Accuracy With Eta = ' + eta + ' and Nodes = ' + nodes)
	plot.show()
