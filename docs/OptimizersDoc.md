# Optimizers.py

## Optimizer

This is the super class that defines a general optimizer

<hr>

<p>
for each property is defined a getter and setter

| Class property        | Explanation                                               |  
| --------------------- | --------------------------------------------------------- |
| type                  | type of the optimizer                                     |
| lr                    | learning rate                                             |
| name                  | name of the optimizer                                     |
| loss_function         | loss function                                             |
| metric                | metric used to check the accuracy                         |
| stop_flag             | a flag for early stopping                                 |
| check_stop            | stopping criteria                                         |
| values_dict           | a dictionary that saves training and validation errors    |
| model                 | model tested                                              | 
| train_dataset         | dataset for training                                      |
| train_labels          | target value                                              |
| batch_size            | batch size                                                |
| pbar                  | dynamic bar visualizer                                    |
| gradient_network      | matrix of units values                                    |
| epoch_training_error  | error computed at each epoch                              |
| epoch_training_error_metric | metric used to compute the training error           |

</p>
<hr>

<p>

<h3>.check_instances</h3>
<p>

<b>Params</b>:

- early_stopping
- check_stop

checks the validity of: <i>early_stopping</i>, <i>self.loss_function</i> and <i>self.metric</i>, and sets <i>
check_stop</i>

</p>
<hr>
<h3> .check_batch </h3>
<p>

<b>Params</b>:

- train_dataset
- train_labels
- batch_size

checks if <i>batch_size</i> is well setted and shapes <i>train_dataset</i> and <i>train_labels</i>

</p>
<hr>
<h3> .set_batches </h3> 
<p>

<b>Params</b>:

- batch_index

sets training_data into batches with given <i>batch_index</i>
</p>
<hr>

<h3> .shuffle_data </h3>
<p>

<b>Params</b>:

- train_dataset
- train_labels

shuffles the <i>training_dataset</i> and <i>train_labels</i>

</p>
<hr>

<h3> .fit_with_gradient </h3>
<p>

<b>Params</b>:

- input
- target

compute weights and biases using gradient with given <i>input</i> and <i>target</i> then apply back propagation
</p>
<hr>
<h3> .compute_training_error </h3> 
<p>
<hr>
computes the training error and saves it in <i>self.value_dict</i>
</p>

<h3> .init_epoch_training_error </h3>
initialize <i>self.training_error</i> and <i>self.training_erorr_metric</i> for each epoch
<hr>
<h3> .validate </h3>
<p>

<b>Param</b>:

- validation

compute <i>validation</i> error
</p>
<hr>
<h3> .apply_stopping </h3>
<p>

<b>Param</b>:

- current_loss_error
- current_val_error apply the stopping criteria to <i>current_loss_error</i> and <i>current_val_error</i>

</p>
<hr>
<h3> .do_epochs </h3>
<p>

<b>Params</b>:

- validation
- epochs
- shuffle
- early_stopping
- optimizer

do training epochs to the model with the given <i>optimizer</i>, checks if <i>shuffle</i> datasets and if <i>
early_stopping</i> is applied
</p>
<hr>

# StochasticGradientDescent extends Optimizer

This class computes the stochastic gradient descent to optimize weights and biases

<hr>
<h3>Properties</h3>
<p>

| Class properties | Explanation |
| --- | --- |
| nesterov | a flag to tell if apply the nesterov method |
| momentum | momentum |
| momentum_network | matrix of the momentum |
| partial_momentum_network | temporary (partial) matrix of the momentum |

</p>
<hr>
<p>

<h3> .apply_nesterov </h3>
<p>
apply nesterov method for stochastic gradient descend
</p>
<hr>

<h3> .apply SGD </h3> 
<p>
apply stochastic gradient descend
</p>
<hr>

<h3> .apply</h3> 
<p>

<b>Param</b>:

- args

apply SGD and nesterov, <i>args</i> is support parameter not used in SGD
</p>

<b>Params</b>:
apply stochastic gradient descend
</p>
<hr>

<h3> .init_SGDnetwork_with_model </h3> 
<p>

<b>Params</b>:

- model

initialize <i>model</i> and SGD matrices to optimize

</p>
<hr>

<h3>.optimization_process</h3>
<p>

<b>Param</b>:

- model
- train_dataset
- train_labels
- epochs
- batch_size
- shuffle
- validation
- early_stopping
- check_stop

process to apply SGD method to train the model

</p>

# RMSProp extends Optimizer

This class computes the gradient descend using RMSProp approach

<hr>
<h3>Properties</h3>
<p>

| Class properties | Explanation |
| --- | --- |
| rho | user defined value |
| epsilon | user defined value |
| rmsprop_network | matrices for computing RMSProp |

</p>

<h3>.init_rms_network_with_model</h3>
<p>

<b>Param</B>:

- model

initialize <i>model</i> and RMSPROP matrices to optimize
<hr>
</p>

<h3>.apply_rms</h3>
<p>

<b>Param</B>:

- args

apply RMSPROP method for training model. <i>args</i> is a support param, not used here
</p>
<hr>
<h3>.optimization_process</h3>
<p>

<b>Param</b>:

- model
- train_dataset
- train_labels
- epochs
- batch_size
- shuffle
- validation
- early_stopping
- check_stop

process to apply RMSPROP method to train the model

</p>

# Adam extends Optimizer

This class computes the gradient descend using ADAM approach

<hr>
<h3>Properties</h3>
<p>

| Class properties | Explanation |
| --- | --- |
| beta1 | user defined value |
| beta2 | user defined value |
| epsilon | user defined value |
| momentum_network_1 | matrices for computing RMSProp |
| momentum_network_2 | matrices for computing RMSProp | 

</p>
<hr>

<h3>.init_Adam_network_with_model</h3>
<p>

<b>Params</b>:

- model

initialize <i>model</i> and ADAM matrices to optimize
</p>

<hr>
<h3>.apply_adam</h3>
<p>

<b>Param</b>:

- batch_index

apply adam to model
</p>
<hr>
<h3>.optimization_process</h3>
<p>

<b>Params</b>:

- model
- train_dataset
- train_labels
- epochs
- batch_size
- shuffle
- validation
- early_stopping
- check_stop

process to apply ADAM method to train the model

</p>


