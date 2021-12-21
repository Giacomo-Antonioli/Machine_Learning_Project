# Model_selection.py

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

</p>

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

The idea behind k-fold cross-validation is to divide all the available data items into roughly equal-sized sets. Each
set is used exactly once as the test set while the remaining data is used as the training set.

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

| Method property                       | Explanation                    |  
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

Function fitting is the process of training a neural network on a set of inputs in order to produce an associated set of
target outputs.
</p>