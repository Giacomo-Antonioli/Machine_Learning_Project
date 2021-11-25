# Model.py

<p>
<h2>
Classes documentation list:
</h2>
<h3>

0. [Home](README.md)
1. [Main.py](./mainDoc.md) 
2. [Model.py](./ModelDoc.md)
3. [Model_selection.py](./model_selectionDoc.md)
4. [Layer.py](./layerDoc.md)
5. [Optimizer.py](./OptimizersDoc.md)
6. [Metrics.py](./metricsDoc.md)
7. [LoadCVSData.py](./loadCSVDataDoc.md)

</h3>

</p>

## Model

The following are methods in the Model.py explained

<h3>Model variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                       | Explanation                                              |  
| ------------------------------------- | -------------------------------------------------------- |
| self.__name = name                    | Name to identify the model                               |
| self.__layers = []                    | Array of layers that form the model                      |     
| self.__optimizer = None               | What optimizer this model uses                           |
| self.__loss = None                    | If this model has loss                                   |
| self.__metrics = None                 | What optimizer this model uses                           |
| self.__training_data = None           | The training data this model uses                        |
| self.__training_targets = None        | What is this model training target                       |
| self.__validation_data = None         | The data for the validation                              |
| self.__epochs = 1                     | The number times the entire training dataset will be used|
| self.__batch_size = None              | Number of samples that will be propagated                |
| self.__validation_split = 0           | Data to be used as validation data                       |
| self.__dense_configuration = []       | The Dense layer                                          |
| self.__net_configuration_types = []   |                                                          |
| self.__early_stopping = False         | If the model has to have an early stop                   | 
| self.__check_stop = None              |                                                          |
| self.__input_shape = 0                | Shape of the input data                                  |               


</p>

<hr>
<h3>add</h3>
<p>
Params: (self, object)

This methods adds a new layer to the layers. It checks if the new layer is a dense layer as well; if it is adds the layer to the dense layers as well. 
</p>

<hr>
<h3>get_empty_struct</h3>
<p>
Params: (self)
Returns a zeroed structure with the same topology of the NN to contain all the layers' gradients.
</p>

<hr>
<h3>forward</h3>
<p>
Params: (self, net_input, training)

| Param             |                                |
|----               |----                            |
|param net_input    | Net's input vector/matrix      |
|training           | Is currently training (?)      |

Performs a forward pass on the whole Network.
returns a net's output vector/matrix
</p>

<hr>
<h3>compile</h3>
<p>
Params: (self, optimizer='sgd', loss='squared', metrics='mee', early_stopping=False, patience=3, tolerance=1e-2,
                monitor='loss', mode='growth')

- param opt: ('Optimizer' object)
- loss: (str) the type of loss function
- metric: (str) the type of metric to track (accuracy etc)

- monitor:                        (?)        
- mode:           (?)         

Prepares the network for training by assigning an optimizer to it and setting its parameters
</p>

<hr>
<h3>fit</h3>
<p>
Params: (self, training_data, training_targets, validation_data=None, epochs=1, batch_size=None, validation_split=0,
            shuffle=False)

- training_data: (numpy ndarray) input training set
- training_targets: (numpy ndarray) targets for each input training pattern
- batch_size: (integer) the size of the batch
- epochs: (integer) number of epochs
- val_split: percentage of training data to use as validation data (alternative to val_x and val_y)
- shuffle: 

Execute the training of the network
</p>

<hr>
<h3>predict</h3>
<p>
Params: (self, prediction_input)

- prediction_input: batch of input patterns                  

 Computes the outputs for a batch of patterns, useful for testing w/ a blind test set.
 Returns an array of net's outputs
</p>

<hr>
<h3>evaluate</h3>
<p>
Params: (self, validation_data, targets, metric=None, loss=None)

- targets: the targets for the input on which the net is evaluated
- metric: the metric to track for the evaluation
- loss: the loss to track for the evaluation


 Performs an evaluation of the network based on the targets and either the pre-computed outputs ('net_outputs')
 or the input data ('net_input'), on which the net will first compute the output.
 If both 'predicted' and 'net_input' are None, an AttributeError is raised

Returns the loss and the metric
</p>

<hr>
<h3>propagate_back</h3>
<p>
Params: (self, dErr_dOut, gradient_network)

- dErr_dOut: derivatives of the error wrt the outputs
- gradient_network: a structure with the same topology of the neural network in question, but used to store the gradients. It will be updated and returned back to the caller

Propagates back the error to update each layer's gradient, returns the updated grad_net

</p>