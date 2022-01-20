# Model_selection.py

<p>
<h2>
Classes documentation list:
</h2>
<h3>

0. <a href="https://giacomo-antonioli.github.io/Machine_Learning_Project/"> Home </a>
1. [Main.py](./mainDoc.md) 
2. [Model.py](./ModelDoc.md)
3. [Model_selection.py](./model_selectionDoc.md)
4. [Layer.py](./layerDoc.md)
5. [Optimizer.py](./OptimizersDoc.md)
6. [Metrics.py](./metricsDoc.md)
7. [LoadCVSData.py](./loadCSVDataDoc.md)
8. [Activations.py](./activations.md)
9. [Loss.py](./loss.md)
10. [Monk.py](./monk.md)
11. [Reguralizers.py](./reguralizers.md)
12. [Weight_Initializer.py](./weightInizializer.md)

</h3>

</p>

# Model Selection
<p>

Model selection is the process of selecting one final machine learning model from among a collection of candidate machine learning models for a training dataset.

Model selection is a process that can be applied both across different types of models (e.g. logistic regression, SVM, KNN, etc.) and across models of the same type configured with different model hyperparameters (e.g. different kernels in an SVM).

</p>

## ValidationTechnique

<p>
Using proper validation techniques helps you understand your model, but most importantly, estimate an unbiased generalization performance.
There is no single validation method that works in all scenarios. It is important to understand if you are dealing with groups, time-indexed data, or if you are leaking data in your validation procedure.
</p>

### SimpleHoldout extends ValidationTechnique

Hold-out is when you split up your dataset into a ‘train’ and ‘test’ set. The training set is what the model is trained on, and the test set is used to see how well that model performs on unseen data. A common split when using the hold-out method is using 80% of data for training and the remaining 20% of the data for testing.

### KFold extends ValidationTechnique

The idea behind k-fold cross-validation is to divide all the available data items into roughly equal-sized sets. Each
set is used exactly once as the test set while the remaining data is used as the training set.

## HyperparametersSearch

<p>
Most common learning algorithms feature a set of hyperparameters that must be determined before training commences. The choice of hyperparameters can significantly affect the resulting model's performance, but determining good values can be complex; hence a disciplined, theoretically sound search strategy is essential.
</p>
<hr>

### GridSearch extends HyperparametersSearch

<p>
Grid search is an approach to hyperparameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.
</p>

---

<h3> References </h3>

- <a href="https://machinelearningmastery.com/a-gentle-introduction-to-model-selection-for-machine-learning/#:~:text=Model%20selection%20is%20the%20process%20of%20selecting%20one%20final%20machine,SVM%2C%20KNN%2C%20etc.)"> https://machinelearningmastery.com/a-gentle-introduction-to-model-selection-for-machine-learning/#:~:text=Model%20selection%20is%20the%20process%20of%20selecting%20one%20final%20machine,SVM%2C%20KNN%2C%20etc.) </a>

- <a href="https://www.datacamp.com/community/tutorials/parameter-optimization-machine-learning-models"> https://www.datacamp.com/community/tutorials/parameter-optimization-machine-learning-models </a>

- <a href="https://www.researchgate.net/publication/272195620_Hyperparameter_Search_in_Machine_Learning"> https://www.researchgate.net/publication/272195620_Hyperparameter_Search_in_Machine_Learning </a>

- <a href="https://medium.com/@eijaz/holdout-vs-cross-validation-in-machine-learning-7637112d3f8f"> https://medium.com/@eijaz/holdout-vs-cross-validation-in-machine-learning-7637112d3f8f </a>

- <a href="https://towardsdatascience.com/validating-your-machine-learning-model-25b4c8643fb7"> https://towardsdatascience.com/validating-your-machine-learning-model-25b4c8643fb7 </a>