# Optimizers.py

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

## Optimizer
 
Optimizers are algorithms or methods used to change the attributes of the neural network such as weights and learning rate to reduce the losses. Optimizers are used to solve optimization problems by minimizing the function.

<hr>

# StochasticGradientDescent extends Optimizer
<p>
SGD algorithm is an extension of the GD algorithm and it overcomes some of the disadvantages of the GD algorithm. GD algorithm has a disadvantage that it requires a lot of memory to load the entire dataset of n-points at a time to computer derivative. In the case of the SGD algorithm derivative is computed taking one point at a time.
 
## Advantage:
1. Memory requirement is less compared to the GD algorithm as derivative is computed taking only 1 point at once.

## Disadvantages:
1. The time required to complete 1 epoch is large compared to the GD algorithm.
2. Takes a long time to converge.
3. May stuck at local minima.
</p>

# RMSProp extends Optimizer

This normalization balances the step size (momentum), decreasing the step for large gradients to avoid exploding and increasing the step for small gradients to avoid vanishing.
Simply put, RMSprop uses an adaptive learning rate instead of treating the learning rate as a hyperparameter. This means that the learning rate changes over time.

# Adam extends Optimizer

<p>

Adam (Kingma & Ba, 2014) is a first-order-gradient-based algorithm of stochastic objective functions, based on adaptive estimates of lower-order moments. Adam is one of the latest state-of-the-art optimization algorithms being used by many practitioners of machine learning. The first moment normalized by the second moment gives the direction of the update.

</p>

---

<h3> References </h3>

- <a href="https://towardsdatascience.com/overview-of-various-optimizers-in-neural-networks-17c1be2df6d5"> https://towardsdatascience.com/overview-of-various-optimizers-in-neural-networks-17c1be2df6d5 </a>

- <a href="https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be#:~:text=RMSprop%20is%20a%20gradient%2Dbased,used%20in%20training%20neural%20networks.&text=This%20normalization%20balances%20the%20step,small%20gradients%20to%20avoid%20vanishing."> https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be#:~:text=RMSprop%20is%20a%20gradient%2Dbased,used%20in%20training%20neural%20networks.&text=This%20normalization%20balances%20the%20step,small%20gradients%20to%20avoid%20vanishing.</a>
