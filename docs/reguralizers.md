# Reguralizers.py

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

<p>

## Reguralizers 

Regularization is a technique which makes slight modifications to the learning algorithm such that the model generalizes better. This in turn improves the model’s performance on the unseen data as well.

Regularization is the most used technique to penalize complex models in machine learning, it is deployed for reducing overfitting (or, contracting generalization errors) by putting network weights small. Also, it enhances the performance of models for new inputs.



</p>

## Lasso L1 (Least Absolute Shrinkage and Selection Operator)
<h3> Description </h3>
<p> 
 
L1 regularization is the preferred choice when having a high number of features as it provides sparse solutions. Even, we obtain the computational advantage because features with zero coefficients can be avoided.


<h3> Pro </h3>

- it is quite capable of reducing the variability and improving the accuracy of linear regression models.

<h3> Cons </h3>

- If the number of predictors (p) is greater than the number of observations (n), Lasso will pick at most n predictors as non-zero, even if all predictors are relevant (or may be used in the test set). In such cases, Lasso sometimes really has to struggle with such types of data.
- If there are two or more highly collinear variables, then LASSO regression select one of them randomly which is not good for the interpretation of data.

</p>

## Ridge L2
<h3> Description </h3>
<p> 
 
 The main algorithm behind this is to modify the RSS by adding the penalty which is equivalent to the square of the magnitude of coefficients. However, it is considered to be a technique used when the info suffers from multicollinearity (independent variables are highly correlated). In multicollinearity, albeit the smallest amount squares estimates (OLS) are unbiased, their variances are large which deviates the observed value faraway from truth value. By adding a degree of bias to the regression estimates, ridge regression reduces the quality errors. It tends to solve the multicollinearity problem through shrinkage parameter λ.


<h3> Pro </h3>

- It decreases the complexity of a model but does not reduce the number of variables since it never leads to a coefficient tending to zero rather only minimizes it

<h3> Cons </h3>

- This model is not a good fit for feature reduction

</p>

---

## Early Stopping
<h3> Description </h3>
<p> 
 
 Early stopping is a method that allows you to specify an arbitrarily large number of training epochs and stop training once the model performance stops improving on the validation dataset.


The challenge is to train the network long enough that it is capable of learning the mapping from inputs to outputs, but not training the model so long that it overfits the training data.



---

<h3> References </h3>

- <a href="https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/"> https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/ </a>

- <a href="https://www.analyticssteps.com/blogs/l2-and-l1-regularization-machine-learning"> https://www.analyticssteps.com/blogs/l2-and-l1-regularization-machine-learning </a>

- <a href="https://www.excelr.com/blog/data-science/regression/l1_and_l2_regularization"> https://www.excelr.com/blog/data-science/regression/l1_and_l2_regularization </a>

- <a href="https://medium.com/zero-equals-false/early-stopping-to-avoid-overfitting-in-neural-network-keras-b68c96ed05d9"> https://medium.com/zero-equals-false/early-stopping-to-avoid-overfitting-in-neural-network-keras-b68c96ed05d9 </a>

- <a href="https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/"> https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/ </a>

