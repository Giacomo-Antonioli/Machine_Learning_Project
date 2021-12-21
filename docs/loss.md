# Loss.py

<p>
<h2>
Classes documentation list:
</h2>
<h3>

0. <a href="https://giacomo-antonioli.github.io/Machine_Learning_Project/"> Home </a>
1. [Main.py](./docs/mainDoc.md) 
2. [Model.py](./docs/ModelDoc.md)
3. [Model_selection.py](./docs/model_selectionDoc.md)
4. [Layer.py](./docs/layerDoc.md)
5. [Optimizer.py](./docs/OptimizersDoc.md)
6. [Metrics.py](./docs/metricsDoc.md)
7. [LoadCVSData.py](./docs/loadCSVDataDoc.md)
8. [Activations.py](./docs/activations.md)
9. [Loss.py](./docs/loss.md)
10. [Monk.py](./docs/monk.md)
11. [Reguralizers.py](./docs/reguralizers.md)
12. [Weight_Initializer.py](./docs/weightInizializer.md)

</h3>

</p>

## Loss
<p>
The Loss Function is one of the important components of Neural Networks. Loss is nothing but a prediction error of Neural Net. And the method to calculate the loss is called Loss Function.
In simple words, the Loss is used to calculate the gradients. And gradients are used to update the weights of the Neural Net. This is how a Neural Net is trained.
</p>

## Binary Crossentropy
<h3> Description </h3>
<p> 
 
Cross entropy is a measure of the difference between two probability distributions. In a machine learning setting using maximum likelihood estimation, we want to calculate the difference between the probability distribution produced by the data generating process (the expected outcome) and the distribution represented by our model of that process.

Binary Cross-Entropy
As the name implies, the binary cross-entropy is appropriate in binary classification settings to get one of two potential outcomes.


<h3> Pro </h3>

- The binary cross-entropy is appropriate in conjunction with activation functions such as the logistic sigmoid that produce a probability relating to a binary outcome.

<h3> Cons </h3>

- Sigmoid is the only activation function compatible with the binary crossentropy loss function.

</p>

## Categorical cross entropy
<h3> Description </h3>
<p> 
 
The categorical cross-entropy is applied in multiclass classification scenarios. In the formula for the binary cross-entropy, we multiply the actual outcome with the logarithm of the outcome produced by the model for each of the two classes and then sum them up. For categorical cross-entropy, the same principle applies, but now we sum over more than two classes. 

The categorical cross-entropy is appropriate in combination with an activation function such as the softmax that can produce several probabilities for the number of classes that sum up to 1.


<h3> Pro </h3>

- This loss is a very good measure of how distinguishable two discrete probability distributions are from each other. 

<h3> Cons </h3>

- Softmax is the only activation function recommended to use with the categorical cross entropy loss function.

</p>

## Squared loss
<h3> Description </h3>
<p> 
 
Squared loss is a loss function that can be used in the learning setting in which we are
predicting a real-valued variable y given an input variable x


</p>

---

<h3> References </h3>

- <a href="https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions"> https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions </a>

- <a href="https://programmathically.com/an-introduction-to-neural-network-loss-functions/"> https://programmathically.com/an-introduction-to-neural-network-loss-functions/ </a>

- <a href="http://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/lossfunctions/SquaredLoss.pdf"> http://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/lossfunctions/SquaredLoss.pdf </a>