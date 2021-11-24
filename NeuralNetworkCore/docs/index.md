# ML Project Notes

The following file contains notes regarding the classes created for the ML Project 2021/2022.

<h2>

1. [Main.py](\mainDoc.md)
2. [Model.py](\Modeldoc.md)

</h2>


# Model_selection.py

## ValidationTechnique

<h3> Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                       | Explanation                                              |  
| ------------------------------------- | ----------------------------   |
| self.__name = name                    | Name to identify the model     |
| self.__training_set = []              |                                |     
| self.__validation_set = []            |                                |
| self.__test_set = []                  |                                |
| self.__training_set_labels = []       |                                |
| self.__validation_set_labels = []     |                                |
| self.__test_set_labels = []           |                                |

</p>

### SimpleHoldout extends ValidationTechnique
<hr>
<h3>split</h3>
<p>
Params: (self, *args)

 Split data in 3 parts (50% training, 25% validation, 25% test-set or with a percentage used defined)
</p>

<hr>
<h3>double_split</h3>
<p>
Params: (self, data, labels)

  Split dataset into training and validation with a 65%-35% split.
</p>

### KFold extends ValidationTechnique

The idea behind k-fold cross-validation is to divide all the available data items into roughly equal-sized sets. Each set is used exactly once as the test set while the remaining data is used as the training set.

<hr>
<h3>split</h3>
<p>
Params: (self, data, splits=5)

 Split the data in splits sets 
</p>

## HyperparametersSearch

<h3>Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                       | Explanation                                              |  
| ------------------------------------- | ----------------------------   |
| self.__name = name                    | Name to identify the model     |
| self.__best_val = None                |                                |     
| self.__best_parameters = []           |                                |
| self.__history = {}                   |                                |
| self.__best_model = None              |                                |


</p>
<hr>

### GridSearch extends HyperparametersSearch


<h3>Class variables</h3>
<p>
For each variable there is a getter and setter.

| Method property                       | Explanation                                              |  
| ------------------------------------- | ----------------------------   |
| self.__model = model                  | \\                             |
| self.__param_list = param_list        |                                |     
| self.__training_set = []              | \\                             |
| self.__validation_set = []            | \\                             |
| self.__evaluated_optimizer            |                                |
| self.__optimizer_seen = False         |                                |
| self.__temp_suspended = {}            |                                |
| self.__current_loss                   | \\                             |
| self.__current_metric                 | \\                             |
| self.__all_reg_mode = False           |                                |
| self.__reguralizers = {}              |                                |
| self.__es = False                     |                                |
| self.__monitor                        | \\                             |
| self.__es_mode                        |                                |
| self.__patience                       | \\                             |
| self.__tol                            |                                |
| self.__epochs                         | \\                             |
| self.__batch_size                     | \\                             |
| self.__shuffle                        |                                |
| self.__cv = 3                         |                                |
| self.__best_val = None                |  \\                            |
| self.__best_params = None             |  \\                            |
| self.__best_tr_metric = None          |  \\                            |
| self.__best_tr_loss = None            |  \\                            |
| self.__best_val_metric = None         |  \\                            |

</p>

<hr>
<h3>add_optimizer_parameters</h3>
<p>
Params: (self, param_combination, param, x)

- param_combination:
- param
- x

(?) 
</p>

<hr>
<h3>fit</h3>
<p>
Params: (self, training_data, training_targets, epochs=None, batch_size=None, shuffle=None, cv=3,
            filename='./curr_dataset')

- training_data:
- training_targets
- epochs=None
- batch_size=None
- shuffle=None
- cv
- filename='./curr_dataset'

Function fitting is the process of training a neural network on a set of inputs in order to produce an associated set of target outputs. 
</p>

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

In any neural network, a dense layer is a layer that is deeply connected with its preceding layer which means the neurons of the layer are connected to every neuron of its preceding layer. This layer is the most commonly used layer in artificial neural network networks.

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

- upstream_delta: for hidden layers, delta = dot_prod(delta_next, w_next) * dOut_dNet
            Multiply (dot product) already the delta for the current layer's weights in order to have it ready for the
            previous layer (that does not have access to this layer's weights) that will execute this method in the
            next iteration of Network.propagate_back()

Sets the layer's gradients
    returns new_upstream_delta: delta already multiplied (dot product) by the current layer's weights
    returns gradient_w: gradient wrt weights
    returns gradient_b: gradient wrt biases

</p>

### Dropout extends Layer

Dropout is an approach to regularization in neural networks which helps reducing interdependent learning amongst the neurons.

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

# Metrics.py

<h3>binary_class_accuracy</h3>
<p>
Params: (predicted, target)

Applies a threshold for computing classification accuracy (correct classification rate).
    If the difference in absolute value between predicted - target is less than a specified threshold it considers it
    correctly classified (returns 1). Else returns 0

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

The true positive rate is calculated as the number of true positives divided by the sum of the number of
    true positives and the number of false negatives.
    It describes how good the model is at predicting the positive class when the actual outcome is positive.

return: tpr, fpr
        WHERE
        tpr is the true positive rate
        fpr is the false positive rate

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

The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value
    of the absolute error loss or l1-norm loss.
    </p>

<hr>
<h3>r2_score</h3>
<p>
Params: (predicted, target)

- predicted: ndarray of shape (n, m) – Predictions for the n examples
- target: ndarray of shape (n, m) – Ground truth for each of n examples

The r2_score function computes the coefficient of determination, usually denoted as R².
    It represents the proportion of variance (of y) that has been explained by the independent variables in the model.
    It provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely
    to be predicted by the model, through the proportion of explained variance.
    As such variance is dataset dependent, R² may not be meaningfully comparable across different datasets.
    Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
    A constant model that always predicts the expected value of y, disregarding the input features, would get a R² score
    of 0.0.
</p>

# LoadCSVData.py
 This class contains the methods to read and split the .csv Datasets and eventually save the new datasets to file
        
<h2> Methods </h2>
   
- loadCSV(path, file_name, size_for_split, save_to_file, save_to_file_path, column_names, column_for_label):  
- saveNewFile(path, file_name, df):                                                                      
- printSets(X_train, X_test):

<h3>loadCSV</h3>
<p>
The method loadCSV recives a csv file and splits it in testing and training DataFrames.

Params: (path, file_name, size_for_split, save_to_file, save_to_file_path, column_names, column_for_label)
- path: the path for the csv file.
- file_name: .csv file name.
- size_for_split: the float value to indicate the size for the test DataFrame 1 = 100%, 0.9 = 90% and so on.
- save_to_file: a boolena value to indicate if the splitted DataSet has to be saved into separated files. (True to save - False not to save)
- save_to_file_path: the path to save the new separated DataFrames (None no specific path - path string for specific path)  
- column_names: list of names for the .csv file columns. 
                    If the number of names are less then the columns or is None the code will assign a standard name equal to c+"number of column"  
- column_for_label: the identifier for the label(s) column(s)
                    The input can be both the numerical index(es) or the string name(s) for the column(s)      
        
Return: The splitted DataSets as different Dataframes. X_test and X_train for the actual data, y_test and y_train for the labels.
</p>

<hr>

<h3>saveNewFile</h3>
<p>
The method saveNewFile is used to save a DataFrame as a new .csv file with a new name in a speific path. 
If the path or the name are null, the code will assign a standard path and name

Params: (path, file_name, df)
- path: the path to save the new csv file.
- file_name: the name for the new .csv file.
- df: the dataFrame from which the new .csv file is created

</p>

<hr>

<h3>printSets</h3>
<p>
The method printSets is used to print a DataFrame in console.

Params: (X_train, X_test)
- X_train: the train portion of the .csv file  
- X_test: the test portion of the .csv file 

</p>


# Glossary

- Epochs: The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
- Batch size: The batch size defines the number of samples that will be propagated through the network.
- Validation split: Fraction of the training data to be used as validation data.
- Dense configuration: The Dense layer is a normal fully connected layer in a neuronal network


