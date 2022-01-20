
from numpy import array
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os.path
from sklearn.preprocessing import OneHotEncoder



class LoadCSVData:
    """
        This class contains the methods to read and split the .csv Datasets
        
        Methods
        ----      
        loadCSV(path, file_name, size_for_split, save_to_file, save_to_file_path, column_names, column_for_label, returnFit):  \n
        saveNewFile(path, file_name, df):                                                                           \n
        printSets(X_train, X_test):                                                                                 \n
    """
    #columns = None
    #monk_dataset, monk_labels= LoadCSVData.loadCSV(path = "datasets/cup/", file_name = 'ML-CUP21-TR.csv', separator=',', column_names=columns, column_for_label=10, drop_rows=[0,1,2,3,4,5,6], drop_cols = [11])
     
    def loadCSV(path, file_name, size_for_split = 1, separator = ',',save_to_file = False, save_to_file_path = None, column_names = None, column_for_label = None, returnFit = False, drop_rows = [], drop_cols = [], shuffle_split = False):

        """
        The method loadCSV recives a csv file and splits it in testing and training DataFrames
        :param path: the path for the csv file.
        :param file_name: the name for the .csv file.
        :param size_for_split: the float value to indicate the size for the test DataFrame 1 = 100%, 0.9 = 90% and so on.
        :param save_to_file: a boolean value to indicate if the splitted DataSet has to be saved into separated files. (True to save - False not to save)
        :param save_to_file_path: the path to save the new separated DataFrames (None no specific path - path string for specific path)  
        :param column_names: list of names for the .csv file columns. 
                    If the number of names are less then the columns or is None the code will assign a standard name equal to c+"number of column"  
        :param column_for_label: the identifier for the label(s) column(s)
                    The input can be both the numerical index(es) or the string name(s) for the column(s)      
        :param returnFit: a boolean value to determin if the X_train and X_test value are to be returned as an array
        :param drop_rows: an array of int that indicate the rows to drop by id
        :param drop_cols: an array of int that indicate the columns to drop by id
        :param shuffle_split: if true the values of the csv file will be shuffled before splitting
        :return: The splitted DataSets as different Dataframes. X_test and X_train for the actual data, y_test and y_train for the lables
    """

        rows = pd.read_csv(path+file_name, skiprows = drop_rows, sep=separator) #Reads the csv file
        df = pd.DataFrame(rows)                     #Create DataFrame                       
        
        if size_for_split == None:
            size_for_split = 0.5

        if column_names != None:
            columns_id = list(column_names)
        else:
            columns_id = list()
            columns_id.append('Id')
            c = 0
            while len(df.columns) < c:
                columns_id.append("c" + str(c))
                c += 1

        
            
        #defines the header names for the csv file if the column names in input are less then the csv column count
        if(len(df.columns) - 1 > len(columns_id)):
            c = len(columns_id) 
            while(len(df.columns) > len(columns_id)):
                columns_id.append("c"+str(c))
                c += 1
                
        #adding the headers to the dataframe
        df = pd.read_csv(path+file_name, skiprows = drop_rows, sep=separator, names = columns_id)        
        df.set_index('Id', inplace=True)
       
        df.drop(df.columns[drop_cols], axis=1, inplace=True) 

        """The following block of code checks if the value for the labels have been passed as an int 
            if the labels are passed as int the code looks for the string version in the column_names
            and passes the int index to y for the label selection and the string version for the x to make the drop
            of the label column(s)
        """
        column_for_y_index = list()
        column_for_drop_index = ""
        if isinstance(column_for_label, int):     
            if drop_cols != None:
                column_for_label -= len(drop_cols)                    
            column_for_y_index = column_for_label
            i = 0
            for column_label in columns_id:
                if i == column_for_label:
                    column_for_drop_index = column_label
                i += 1
        else:
            i = 0
            for column_label in columns_id:
                if column_label == column_for_label:
                    column_for_y_index = i
                i += 1
            column_for_drop_index = column_for_label

        if column_for_y_index == None or column_for_y_index == []:
            y = df.iloc[:, 0]
            X = df

        else:     
            y = df.iloc[:,column_for_y_index] 
            X = df.drop(column_for_drop_index, axis=1)                          #the dataframe except for the labels; axis is 1 for columns, 0 for rows           
             
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=size_for_split, shuffle = shuffle_split) #splits the dataset in test and train
        y_train = y_train.to_list()
        y_train = np.reshape(y_train, (len(y_train), 1)) 
        y_test = y_test.to_list() 
        y_test = np.reshape(y_test, (len(y_test), 1))
        
        if(save_to_file):
            LoadCSVData.saveNewFile(save_to_file_path,"X-train",X_train)
            LoadCSVData.saveNewFile(save_to_file_path,"X-test",X_test)
            LoadCSVData.saveNewFile(save_to_file_path,"y-train",y_train)
            LoadCSVData.saveNewFile(save_to_file_path,"y-test",y_train)
 
        if size_for_split == 1 and returnFit == False:
            return X_train.to_numpy().astype(np.float32), y_train.astype(np.float32)
        if size_for_split == 1:
            X_train = OneHotEncoder().fit_transform(X_train).toarray().astype(np.float32)
            return X_train, y_train
        if returnFit == False:
            return  X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.astype(np.float32), y_test.astype(np.float32)
        else:
             X_train = OneHotEncoder().fit_transform(X_train).toarray().astype(np.float32)
             X_test = OneHotEncoder().fit_transform(X_test).toarray().astype(np.float32)
             return  X_train, X_test, y_train, y_test
        

    def saveNewFile(path, file_name, df):
        """
        The method saveNewFile is used to save a DataFrame as a new .csv file with a new name in a specific path 
        if the path or the name are null, the code will assign a standard path and name
        
        :param path: the path to save the new csv file.
        :param file_name: the name for the new .csv file.
        :param df: the dataFrame from which the new .csv file is created
        """
        if (path == None):
            path = "datasets"
        if (file_name == None):
            file_name = "splittedSet"

        
        df.to_csv(os.path.join(path, file_name+'.csv'))
        
    
    def printSets(df_name, dataFrame):

        """
        The method printSets is used to print a DataFrame in console.
        :param df_name: the dataframe name 
        :param dataFrame: the dataframe to print 
        """

        print(df_name)
        print(dataFrame)
        print(dataFrame.shape)
        

#the following code is for testing purpose only
#columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
#monk_dataset, monk_labels= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = "monks-1.train", separator=' ', column_names=columns, column_for_label=0, returnFit=True)   
#print("-here-")
#print(monk_labels)
#print(monk_dataset)
#LoadCSVData.loadCSV(path = "/datasets/cup/", file_name = 'ML-CUP21-TR.csv', separator=',', column_for_label=10, drop_cols=['#'])

# the following code is for testing purpose only
# columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13']
# LoadCSVData.loadCSV("datasets\cup","\CUP-INTERNAL-TEST.csv", 0.5)
# LoadCSVData.loadCSV("datasets\cup","\CUP-INTERNAL-TEST.csv", 0.5, True, save_to_file_path=None)