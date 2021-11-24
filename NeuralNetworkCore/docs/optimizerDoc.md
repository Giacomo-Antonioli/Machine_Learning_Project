# Optimizer.py

## Optimizer
Abstract class representing a generic optimizer
| Method property                                    | Explanation                                           |  
| -------------------------------------              | ----------------------------                          |
| self.__type = 'optimizer'                          | \\                                                    |

### StochasticGradientDescent extends Optimizer

Stochastic gradient descent (often abbreviated SGD) is an iterative method for optimizing an objective function with suitable smoothness properties

<h3>Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                                    | Explanation                                                        |  
| -------------------------------------              | ----------------------------                                       |
| self.__lr = learning_rate                          | Learning Rate parameter                                            |
| self.__momentum = momentum                         | \\                                                                 |     
| self.__nesterov = nesterov                         | Bool that indicates the usage of the nesterov momentum technique   |
| self.__name = 'sgd'                                | seed of the randomization                                          |
| self.__loss_function = ''                          | \\                                                                 |                 
| self.__metric = ''                                 | \\                                                                 |                 

Class that represent a dense layer of a neural network
</p>

<hr>
<h3>optimization_process</h3>
<p>
Params: (self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None)

- model:
- train_dataset
- train_labels
- epochs=1
- batch_size=1
- shuffle=False
- validation=None
- early_stopping=False
- check_stop=None

(?)
</p>

### RMSProp extends Optimizer

 Root Mean Square Propagation
 https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a

<h3>Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                                    | Explanation                                                        |  
| -------------------------------------              | ----------------------------                                       |
| self.__lr = learning_rate                          | Learning Rate parameter                                            |
| self.__momentum = momentum                         | \\                                                                 |     
| self.__rho = rho                                   | (?)                                                                |
| self.__name = 'sgd'                                | seed of the randomization                                          |
| self.__loss_function = ''                          | \\                                                                 |                 
| self.__metric = ''                                 | \\                                                                 |                 
| self.__epsilon=1e-8                                | \\                                                                 |                 

Class that represent a dense layer of a neural network
</p>

<hr>
<h3>optimization_process</h3>
<p>
Params: (self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None)

- model:
- train_dataset
- train_labels
- epochs=1
- batch_size=1
- shuffle=False
- validation=None
- early_stopping=False
- check_stop=None

(?)

</p>

### Adam extends Optimizer

Adaptive Moment Estimation
    https://arxiv.org/abs/1412.6980


<h3>Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                                    | Explanation                                                        |  
| -------------------------------------              | ----------------------------                                       |
| self.__lr = learning_rate                          | Learning Rate parameter                                            |
| self.__momentum = momentum                         | \\                                                                 |     
| self.__rho = rho                                   | (?)                                                                |
| self.__name = 'adam'                               | \\                                                                 |
| self.__loss_function = ''                          | \\                                                                 |                 
| self.__metric = ''                                 | \\                                                                 |                 
| self.__beta1 = beta1                               |                                                                    |                 
| self.__beta2 = beta2                               |                                                                    |                 
| self.__epsilon=epsilon                             |                                                                    |                 

Class that represent a dense layer of a neural network
</p>

<hr>
<h3>optimization_process</h3>
<p>
Params: (self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None)

- model:
- train_dataset
- train_labels
- epochs=1
- batch_size=1
- shuffle=False
- validation=None
- early_stopping=False
- check_stop=None

(?)

</p>

# Glossary

- Epochs: The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
- Batch size: The batch size defines the number of samples that will be propagated through the network.
- Validation split: Fraction of the training data to be used as validation data.
- Dense configuration: The Dense layer is a normal fully connected layer in a neuronal network