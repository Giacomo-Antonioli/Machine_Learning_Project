# Layer.py

## Layer

<h3> Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                       | Explanation                                              |  
| ------------------------------------- | ----------------------------   |
| self.__type = type                    |                                |

Class that represent a layer of a neural network
</p>

### Dense extends Layer

In any neural network, a dense layer is a layer that is deeply connected with its preceding layer which means the
neurons of the layer are connected to every neuron of its preceding layer. This layer is the most commonly used layer in
artificial neural network networks.

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

Dropout is an approach to regularization in neural networks which helps reducing interdependent learning amongst the
neurons.

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