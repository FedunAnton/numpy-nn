import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from Dense import Dense
from Layer import Layer
from Activations import ReLU, Tanh, Sine, Sigmoid
import argparse

np.random.seed(42)
PIK = "weights.dat"


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def compute_sigmoid_loss(logits, reference_answers):
	y_hat = []
	pr = []
	loss = []
	for i in range(0, len(logits)):
		pr.append( softmax(logits[i, :]))
		if reference_answers[i] == 1:
			y_hat.append(pr[-1][1])
		else:
			y_hat.append(pr[-1][0])
	for i in range(0, len(logits)):
		lfa = logits[np.arange(len(logits)), reference_answers][i]
		loss.append(- lfa + np.log(np.sum(np.exp(logits[i]))))
	return loss


def grad_compute_sigmoid_loss(logits, reference_answers):
	logits_comp = np.zeros(logits.shape)
	logits_comp[np.arange(len(logits)), reference_answers] = 1
	softmaxes = []

	for i in range(len(logits)):
		softmaxes.append(softmax(logits[i, :]))
	ret = - logits_comp + softmaxes
	return ret / logits.shape[0]




def data_preparation(df, test_size=0.8):
    y = df[1]
    X = df.drop([1],axis='columns')
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    msk = np.random.rand(len(X)) < test_size
    X_train = X[msk]
    X_val = X[~msk]
    y_train = y[msk]
    y_val = y[~msk]


    X_train = X_train.values #pd.DataFrame(X_train.values)
    y_train = y_train.values.reshape(y_train.shape[0],)# pd.DataFrame(y_train.values)
    X_val = X_val.values # pd.DataFrame(X_val.values)
    y_val = y_val.values.reshape(y_val.shape[0],) # pd.DataFrame(y_val.values)

    return X_train, y_train, X_val, y_val


def forward(network, X):
	activations = []
	inpu = X
	for layer in network:
		activations.append(layer.forward(inpu))
		inpu = activations[-1]

	return activations


def predict(network, X):
	logits = forward(network, X)[-1]
	return logits.argmax(axis=-1)


def train(network, X, y, X_val, y_val):

	layer_activations = forward(network, X)
	layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
	logits = layer_activations[-1]
	loss = compute_sigmoid_loss(logits, y)

	netw2 = network
	layer_activations_val = forward(netw2, X_val)
	logits_val = layer_activations_val[-1]
	loss_val = compute_sigmoid_loss(logits_val, y_val)


	loss_grad = grad_compute_sigmoid_loss(logits, y)
	for layer_index in range(len(network))[::-1]:
		loss_grad = network[layer_index].backward(layer_inputs[layer_index], loss_grad)

	return np.mean(loss), np.mean(loss_val)


def get_minibatches(inputs, targets, batchsize):
    indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def construct_model_default(input_shape, learning_rate=0.1):
	network = []
	network.append(Dense(input_shape, 100, learning_rate=learning_rate))
	network.append(ReLU())
	network.append(Dense(100, 200, learning_rate=learning_rate))
	network.append(Sine())
	network.append(Dense(200, 2, learning_rate=learning_rate))
	return network


def custom_model(input_shape, learning_rate):
	network = []
	try:
		print("Enter the number of layers: ")
		n_layers = int(input())
		prev_input = input_shape
		for i in range(n_layers - 1):
			print("Now, enter the number of neurons for Layer #", i+1)
			neurons = int(input())
			network.append(Dense(prev_input, neurons, learning_rate=learning_rate))
			print("Its time to choose an activation function.")
			print("1 for ReLu, 2 for Tanh, 3 for Sine, 4 for Sigmoid: ")
			activ = int(input())
			if activ == 1:
				network.append(ReLU())
			elif activ == 2:
				network.append(Tanh())
			elif activ == 3:
				network.append(Sine())
			elif activ == 4:
				network.append(Sigmoid())
			else:
				raise Exception

			prev_input = neurons
		network.append(Dense(prev_input, 2, learning_rate=learning_rate))
		return network

	except:
		print("You are doing it wrong! Initializing default model...")
		return construct_model_default(input_shape, learning_rate)



def process_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--epochs", required = False, help = "Number of epochs")
	ap.add_argument("-a", "--alpha", required = False, help = "Training speed")
	ap.add_argument("-c", "--custom", action="store_true", required=False, help="Create custom neural net")
	ap.add_argument("-stop", "--early_stop", required=False, help="Early stopping")
	ap.add_argument("-f", "--file", required = True, help = "Path to dataset")
	ap.add_argument("-b", "--batch_size", required = False, help = "Batch size")
	args = vars(ap.parse_args())
	return args


def plot_loss(train_loss, val_loss):
	plt.plot(train_loss, label='train loss')
	plt.plot(val_loss, label='val loss')
	plt.legend(loc='best')
	plt.grid()
	plt.show()


def plot_acc(train_acc, val_acc):
	plt.plot(train_acc, label='train accuracy')
	plt.plot(val_acc, label='val accuracy')
	plt.legend(loc='best')
	plt.grid()
	plt.show()


def main(args):
	df = pd.read_csv(args['file'], header=None)
	X_train, y_train, X_val, y_val = data_preparation(df, 0.7)
	y_train = (y_train == 'M').astype(int)
	y_val = (y_val == 'M').astype(int)

	train_log, val_log = [], []
	train_loss, val_loss = [], []
	networks = []
	early_stopping_rounds = 20 if not args['early_stop'] else int(args['early_stop'])
	learning_rate = 0.1 if not args['alpha'] else float(args['alpha'])
	n_epochs = 500 if not args['epochs'] else int(args['epochs'])
	batch_size = 200 if not args['batch_size'] else int(args['batch_size'])
	network = construct_model_default(X_train.shape[1], learning_rate) if not args['custom']\
		else custom_model(X_train.shape[1], learning_rate)


	for epoch in range(n_epochs):
		los = []
		val_los = []
		for x_batch, y_batch in get_minibatches(X_train, y_train, batchsize=batch_size):
			l, l1 = train(network, x_batch, y_batch, X_val, y_val)
			los.append(l)
			val_los.append(l1)
		train_loss.append(np.mean(los))
		val_loss.append(np.mean(val_los))

		train_log.append(np.mean(predict(network, X_train) == y_train))
		val_log.append(np.mean(predict(network, X_val) == y_val))
		networks.append(network)
		if early_stopping_rounds <= epoch and early_stopping_rounds != 0:
			if val_loss[epoch] >= val_loss[epoch - early_stopping_rounds]:
				network = networks[epoch - early_stopping_rounds]
				break

		print("Epoch", epoch, " - ", "Train acc:", train_log[-1], " - ", "Val acc:", val_log[-1])
		print("Train loss:", train_loss[-1], " - ", "Val loss:", val_loss[-1])

	plot_acc(train_log, val_log)
	plot_loss(train_loss, val_loss)
	data = network
	with open(PIK, "wb") as f:
		pickle.dump(data, f)
	print("Weights are saved to " + PIK)


if __name__ == "__main__":
	args = process_args()
	try:
		main(args)
	except Exception as e:
		print("You did something wrong! Details:")
		print(e)
