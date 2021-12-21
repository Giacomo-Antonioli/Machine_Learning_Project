# LoadCSVData.py


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

## LoadCSVData

 This class contains the methods to read and split the .csv Datasets and eventually save the new datasets to file
        
=======

<h2> Methods </h2>

- loadCSV(path, file_name, size_for_split, save_to_file, save_to_file_path, column_names, column_for_label):
- saveNewFile(path, file_name, df):
- printSets(X_train, X_test):


<h3>loadCSV</h3>
<p>
The method loadCSV receives a csv file and splits it in testing and training DataFrames.

Params: (path, file_name, size_for_split, save_to_file, save_to_file_path, column_names, column_for_label)

- path: the path for the csv file.
- file_name: .csv file name.
- size_for_split: the float value to indicate the size for the test DataFrame 1 = 100%, 0.9 = 90% and so on.

- save_to_file: a boolean value to indicate if the splitted DataSet has to be saved into separated files. (True to save - False not to save)
- save_to_file_path: the path to save the new separated DataFrames (None no specific path - path string for specific path)  
- column_names: list of names for the .csv file columns. 
                    If the number of names are less then the columns or is None the code will assign a standard name equal to c+"number of column"  
- column_for_label: the identifier for the label(s) column(s)
                    The input can be both the numerical index(es) or the string name(s) for the column(s)      
- returnFit: a boolean value to determine if the X_train and X_test value are to be returned as an array
        
If the size_for_split == 1 and returnFit = False
Return: The DataSets as Dataframe and it's Label. X_train and y_train
If the size_for_split == 1 and returnFit = True
Return: The DataSets as Dataframe and it's Label. X_train as an array and y_train
If the size_for_split != 1 and returnFit = False
Return: The splitted DataSets as different Dataframes. X_test and X_train for the actual data, y_test and y_train for the labels.
If the size_for_split != 1 and returnFit = True
Return: The splitted DataSets as different Dataframes. X_test and X_train for the actual data as an array, y_test and y_train for the labels.

</p>

<hr>

<h3>saveNewFile</h3>
<p>
The method saveNewFile is used to save a DataFrame as a new .csv file with a new name in a specific path. 
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