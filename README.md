# Experiment-3-Implementation-of-MLP-for-non-linear-separable-problem
**AIM:**

To implement a perceptron for classification using Python

**EQUIPMENTS REQUIRED:**
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

**RELATED THEORETICAL CONCEPT:**
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows
XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

**Algorithm :**

Step 1 : Initialize the input patterns for XOR Gate
Step 2: Initialize the desired output of the XOR Gate
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron 
              and 1 output neuron
Step 3: Repeat the  iteration  until the losses become constant and 
              minimum
              (i)  Compute the output using forward pass output
              (ii) Compute the error  
		          (iii) Compute the change in weight ‘dw’ by using backward 
                     propagation algorithm.
             (iv) Modify the weight as per delta rule.
             (v)   Append the losses in a list
Step 4 : Test for the XOR patterns.

** PROGRAM** 
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length','sepal-width','petal-length','petal-width','Class']
irisdata = pd.read_csv(url, names=names)
X=irisdata.iloc[:,0:4]
y= irisdata.select_dtypes(include=[object])
X.head()
y.head()
y.Class.unique()
le = preprocessing.LabelEncoder()
y= y.apply(le.fit_transform)
y.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.20)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train,y_train.values.ravel())
predictions = mlp.predict(X_test)
print(predictions)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


 **OUTPUT** 
 <img width="491" alt="image" src="https://user-images.githubusercontent.com/94155480/232685267-ca1beb21-5da6-4d9a-a5eb-24b6f57356fd.png">


** RESULT**
Thus the program for implementation for MLP executed sucessfully
