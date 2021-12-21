# Activations.py

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

## Activations functions
<p>
An activation function is a function that is added into an artificial neural network in order to help the network learn complex patterns in the data. 
</p>

## relu
<h3> Description </h3>
<p> 
 relu which stands for Rectified Linear Units. The formula is deceptively simple: max(0,z). Despite its name and appearance, it’s not linear and has the following pro and cons:

<h3> Pro </h3>

- It avoids and rectifies vanishing gradient problem.
- ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations.


<h3> Cons </h3>

- One of its limitations is that it should only be used within hidden layers of a neural network model.
- Some gradients can be fragile during training and can die. It can cause a weight update which will makes it never activate on any data point again. In other words, - ReLu can result in dead neurons.
- The range of ReLu is [0,∞). This means it can blow up the activation.

</p>

## leaky relu
<h3> Description </h3>
<p> 
 LeakyRelu is a variant of ReLU. Instead of being 0 when z<0, a leaky ReLU allows a small, non-zero, constant gradient α (Normally, α=0.01). 

<h3> Pro </h3>

- Leaky ReLUs are one attempt to fix the “dying ReLU” problem by having a small negative slope (of 0.01, or so).


<h3> Cons </h3>

- As it possess linearity, it can’t be used for the complex Classification. It lags behind the Sigmoid and Tanh for some of the use cases.

</p>

## sigmoid
<h3> Description </h3>
<p> 
 Sigmoid takes a real value as input and outputs another value between 0 and 1.
 It’s non-linear, continuously differentiable, monotonic, and has a fixed output range.

<h3> Pro </h3>

- It is nonlinear in nature. 
- It will give an analog activation unlike step function.
- It has a smooth gradient too.
- It’s good for a classifier.
- The output of the activation function is always going to be in range (0,1) compared to (-inf, inf) of linear function. So we have our activations bound in a range.


<h3> Cons </h3>

- Towards either end of the sigmoid function, the Y values tend to respond very less to changes in X.
- It gives rise to a problem of “vanishing gradients”.
- Its output isn’t zero centered. It makes the gradient updates go too far in different directions. 0 < output < 1, and it makes optimization harder.
- Sigmoids saturate and kill gradients.
- The network refuses to learn further or is drastically slow.

</p>

## tanh
<h3> Description </h3>
<p> 
Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear. But unlike Sigmoid, its output is zero-centered. 

<h3> Pro </h3>

- The gradient is stronger for tanh than sigmoid ( derivatives are steeper).


<h3> Cons </h3>

- Tanh also has the vanishing gradient problem.

</p>

---

<h3> References </h3>
<a href="https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html"> ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html </a>



