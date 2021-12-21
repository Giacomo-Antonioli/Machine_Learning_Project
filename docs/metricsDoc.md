# Metrics.py

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

## Metrics

<h3>binary_class_accuracy</h3>
<p>
Params: (predicted, target)

Applies a threshold for computing classification accuracy (correct classification rate). If the difference in absolute
value between predicted - target is less than a specified threshold it considers it correctly classified (returns 1).
Else returns 0

</p>

<hr>
<h3>mean_euclidean_error</h3>
<p>
Params: (predicted, target)

- predicted: ndarray of shape (n, m) – Predictions for the n examples
- target: ndarray of shape (n, m) – Ground truth for each of n examples

Computes the euclidean error between the target vector and the output predicted by the net
</p>

<hr>
<h3>true_false_positive</h3>
<p>
Params: (predicted, target)

- predicted: ndarray of shape (n, m) – Predictions for the n examples
- target: ndarray of shape (n, m) – Ground truth for each of n examples

The true positive rate is calculated as the number of true positives divided by the sum of the number of true positives
and the number of false negatives. It describes how good the model is at predicting the positive class when the actual
outcome is positive.

return: tpr, fpr WHERE tpr is the true positive rate fpr is the false positive rate

</p>

<hr>
<h3>mean_euclidean_error</h3>
<p>
Params: (predicted, target)

- predicted: ndarray of shape (n, m) – Predictions for the n examples
- target: ndarray of shape (n, m) – Ground truth for each of n examples

Computes the euclidean error between the target vector and the output predicted by the net
</p>

<hr>
<h3>mean_absolute_error</h3>
<p>
Params: (predicted, target)

- predicted: ndarray of shape (n, m) – Predictions for the n examples
- target: ndarray of shape (n, m) – Ground truth for each of n examples

The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value of the
absolute error loss or l1-norm loss.
</p>

<hr>
<h3>r2_score</h3>
<p>
Params: (predicted, target)

- predicted: ndarray of shape (n, m) – Predictions for the n examples
- target: ndarray of shape (n, m) – Ground truth for each of n examples

The r2_score function computes the coefficient of determination, usually denoted as R². It represents the proportion of
variance (of y) that has been explained by the independent variables in the model. It provides an indication of goodness
of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the
proportion of explained variance. As such variance is dataset dependent, R² may not be meaningfully comparable across
different datasets. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A
constant model that always predicts the expected value of y, disregarding the input features, would get a R² score of
0.0.
</p>