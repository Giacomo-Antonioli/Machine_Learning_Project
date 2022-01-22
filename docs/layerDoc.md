# Layer.py

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

## Layer

Description: The Layers class is the main building block of the given simulator. It implements a single layer of a MLP and specializes into two different types: a Dense layer and a Dropout layer. Due to the fact that the aim for the layers is to receive an input and calculate the output and finally doing the back-propagation to adjust the weights for each unit, the forward_pass and backwards_pass methods are present in both the layers implementation.

### Dense extends Layer

A dense layer is a layer that is deeply connected with its preceding layer which means the neurons of the layer are connected to all the neurons of its preceding layer. It contains the matrix for both weights and biases, based on the input dimension and the number of units contained in this layer. The input dimensions of each layer, except from the first one, are retrieved from the previous dense layer. The weight and biases matrices are initialized using different initialization methods and there are different options from which to choose(as discussed in section Initializer. To the internal propagation of the input through the layer an activation function is applied to the final computational result. Each layer has the possibility to implement some kinds of regularization independently from each other.

<hr>
<h3>Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                                 | Explanation                                              |  
| -------------------------------------           | ----------------------------   |
| self.__weight_initializer=weight_initializer    | \\                             |
| self.__bias_initializer=bias_initializer        | \\                             |     
| self.__weights=[]                               | \\                             |
| self.__biases=[]                                | \\                             |
| self.__input_dimension = 0                      | \\                              |
| self.__n_units = n_units                        | \\                             |
| self.__activation_function                      | \\                             |
| self.__inputs = None                            | \\                             |
| self.__nets = None                              |                                |
| self.__outputs = None                           |                                |
| self.__gradient_w = None                        |                                |
| self.__gradient_b = None                        |                                |
| self.__regularizer = None                       |                              |
| self.__regularizer_param = None                 |                                |

Class that represent a dense layer of a neural network
</p>

<hr>
<h3>forward_pass</h3>
<p>
Params: (self, input)

- input: (numpy ndarray) input vector

Performs the forward pass on the current layer returns the vector of the current layer's outputs
</p>

<hr>
<h3>backward_pass</h3>
<p>
Params: (self, upstream_delta)

- upstream_delta: for hidden layers, delta = dot_prod(delta_next, w_next) * dOut_dNet Multiply (dot product) already the
  delta for the current layer's weights in order to have it ready for the previous layer (that does not have access to
  this layer's weights) that will execute this method in the next iteration of Network.propagate_back()

Sets the layer's gradients returns new_upstream_delta: delta already multiplied (dot product) by the current layer's
weights returns gradient_w: gradient wrt weights returns gradient_b: gradient wrt biases

</p>

### Dropout extends Layer

The dropout layer is a special layer that only changes the input for the next dense layer zeroing some random inputs. It can be initialized with an user selected dropout ratio and with a specific seed for the random generator. This zeroing process is only applied to the feed-forward propagation. 

<hr>
<h3>Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                                    | Explanation                                           |  
| -------------------------------------              | ----------------------------                          |
| self.__original_inputs                             | \\                                                    |
| self.__outputs=[0]                                 | \\                                                    |     
| self.__probability = probability                   | probability percentage of an input to not be passed on|
| self.__seed = seed                                 | seed of the randomization                             |
| self.__rng_generator = np.random.default_rng(seed) | \\                                                    |                 

Class that represent a dense layer of a neural network
</p>