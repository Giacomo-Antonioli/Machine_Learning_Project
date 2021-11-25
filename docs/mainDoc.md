# main.py

The following are methods in the main.py explained

<hr>
<h3>.read_csv</h3>
<p> 
something = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))

cvs = comma separated values pd.read_csv() = Pandas read_csv() function imports a CSV file to DataFrame format.

| Method property      | Explanation                                               |  
| -------------------- | --------------------------------------------------------- |
| directory + file     |                                                           |
| sep=','              | Delimiter to use.                                         |
| names=col_names      | List of column names to use.                              |
| skiprows=range(7)    | Number of lines to skip (int) at the start of the file.   |
| usecols=range(1, 11) | Column(s) to use as the row labels of the DataFrame       |

</p>

<hr>
<h3>.to_numpy</h3>
<p> 
devset_x = tr_data.to_numpy(dtype=np.float32)

Convert the DataFrame to a NumPy array. By default, the dtype of the returned array will be the common NumPy dtype of
all types in the DataFrame.

| Method property      | Explanation                                               |  
| -------------------- | --------------------------------------------------------- |
| dtype=np.float32     | The dtype to pass to numpy.asarry()                       |  

</p>

<hr>
<h3>np.reshape</h3>
<p> 
data_trainy_1 = np.reshape(devset_y[:int(len(devset_y)*0.9), 0], (-1, 1))

Gives a new shape to an array without changing its data.

</p>

<hr>
<h3>Other classes called</h3>
<p> 
The following are other classes used in the main explained later in this file 

- Model
- GridSearch
- Dense
- Dropout
- RMSProp
- GridSearch

</p>
