import pandas as pd
import numpy as np
import argparse
import pickle
from train import predict, forward, compute_sigmoid_loss
np.random.seed(42)


def get_loss(network, X, y):
	layer_activations = forward(network, X)
	logits = layer_activations[-1]
	loss = compute_sigmoid_loss(logits, y)
	return np.mean(loss)


def data_preparation(df, test_size=0.8):
    y = df[1]
    X = df.drop([1], axis='columns')
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()


    X_train = X
    y_train = y

    X_train = X_train.values #pd.DataFrame(X_train.values)
    y_train = y_train.values.reshape(y_train.shape[0],)# pd.DataFrame(y_train.values)

    return X_train, y_train


def F1_score(tags,predicted):
	tp = 0
	fp = 0
	fn = 0
	for i in range(len(tags)):
		if tags[i] == 1 and predicted[i] == 1:
			tp += 1
		if tags[i] == 1 and predicted[i] == 0:
			fp += 1
		if tags[i] == 0 and predicted[i] == 1:
			fn += 1

	precision = float(tp) / (tp + fp)
	recall = float(tp) / (tp + fn)

	return 2 * ((precision * recall) / (precision + recall))


def process_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--file", required=True, help="Path to dataset")
	ap.add_argument("-w", "--weights", required=True, help="Path to weights file")
	args = vars(ap.parse_args())
	return args


def main(args):

	with open(args['weights'], "rb") as f:
		data5 = pickle.load(f)
	df = pd.read_csv(args['file'], header=None)

	X, y = data_preparation(df)
	y = (y == 'M').astype(int)
	loss = get_loss(data5, X, y)
	print("LOSS ", loss)
	y_hat = predict(data5, X)
	print("ACCURACY ", np.mean(y_hat == y))
	print("F1-score ", F1_score(y, y_hat))

	predicted = pd.DataFrame(y_hat)
	predicted.columns = ['Predicted']
	predicted[predicted['Predicted'] == 1] = 'M'
	predicted[predicted['Predicted'] == 0] = 'B'
	predicted.to_csv('predicted.csv')


if __name__ == "__main__":
	args = process_args()
	try:
		main(args)
	except Exception as e:
		print("You did something wrong!")
		print(e)
