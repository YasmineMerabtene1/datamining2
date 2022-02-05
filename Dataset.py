from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import *
from Svm import SVM
dataset = pd.read_csv("Dataset.csv")

x = dataset["age"].tolist()
y = dataset["target"].tolist()
z = pow(dataset['target'], 2) * pow(dataset['age'], 2).tolist()



# affichage
target = 1
no_target = 0



def convert_target(target: int) -> int:

    if target == 1:
        return target
    elif target == 0:
        return no_target

    else:
        raise ValueError('Mauvaise donnée: ' + target)


def convert_data(df) -> list:
    r = []


    for index, line in df.iterrows():
        obs = [
            float(line['age']),
            float(line['sex']),
            float(line['cp']),
            float(line['trestbps']),
            float(line['chol']),
            float(line['fbs']),
            float(line['restecg']),
            float(line['thalach']),
            float(line['exang']),
            float(line['oldpeak']),
            float(line['slope']),
            float(line['ca']),
            float(line['thal']),
            convert_target(line['target'])


        ]

        r.append(obs)




    return r

def accuracy(predictions, trues) -> float:
	total = 0
	for i in range(0, len(predictions)):
		if predictions[i] == trues[i]:
			total += 1
	return total / len(predictions)






def main(filename: str = 'Dataset.csv'):
    # Chargement des données
    df = pd.read_csv(filename)
    print("L'entête du dataframe: \n\n", df)
    # Convertir les données
    data = convert_data(df)
    data = np.array(data)
    # Mélanger les données
    np.random.shuffle(data)

    # Extraction des données
    x = data[:, 0:11]
    y = data[:, 11]
    # Normalisation des données
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / std
    # division des données
    div_index = int(0.8 * len(x))
    x_train = x[0:div_index]
    y_train = y[0:div_index]
    x_test = x[div_index:]
    y_test = y[div_index:]

    clf = SVM()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    acc = accuracy(pred, y_test)
    print('Accuracy  = ', acc)
main()

d = np.array(dataset)
z = np.array(z)
clf = SVM()
clf.fit(d, z)

df = pd.read_csv('Dataset.csv')

    # Convertir les données
data = convert_data(df)
data = np.array(data)
X = []
y1 = []
for l in data:
    X.append(l[:-1])
    y1.append(l[-1])

y1 = np.array(y)
X = np.array(X)
clf1 = SVM()
clf1.fit(X, y1)

def visualize_svm():
	def get_hyperplane_value(x, w, b, offset):
		return (-w[0] * x + b + offset / w[1])
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.scatter(X[:,0], X[:,1], marker='o',c=y)

	x0_1 = np.amin(X[:,0])
	x0_2 = np.amax(X[:, 0])

	x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
	x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

	x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
	x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

	x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
	x1_2_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)

	ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
	ax.plot([x0_1, x0_2],[x1_1_m , x1_2_m])
	ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p])

	x1_min = np.amin(X[:,1])
	x1_max = np.amax(X[:,1])
	ax.set_ylim([x1_min-3,x1_max+3])

	plt.show()
visualize_svm()

def deux_D():

    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset / w[1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(x, z, marker='o', c=z)
    x0_1 = np.amin(x)
    x0_2 = np.amax(x)
    x1_1 = get_hyperplane_value(x0_1, clf1.w, clf1.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf1.w, clf1.b, 0)
    x1_1_m = get_hyperplane_value(x0_1, clf1.w,clf1.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf1.w, clf1.b, -1)
    x1_1_p = get_hyperplane_value(x0_1, clf1.w, clf1.b, 1)
    x1_2_p = get_hyperplane_value(x0_1, clf1.w, clf1.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    
    x1_min = np.amin(z)
    x1_max = np.amax(z)
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()
def trois_D():


    # creating 3d figures
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(x, y, z, marker='o',
                     s=99, c=y)


    # adding title and labels
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()






trois_D()
deux_D()

