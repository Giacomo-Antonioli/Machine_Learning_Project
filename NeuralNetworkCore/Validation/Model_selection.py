#  Copyright (c) 2021.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy as np
from functools import singledispatch


class Validation_Technique:
    def __init__(self, name):
        self.__name = name
        self.__training_set = []
        self.__validation_set = []
        self.__test_set = []

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def training_set(self):
        return self.__training_set

    @training_set.setter
    def training_set(self, training_set):
        self.__training_set = training_set

    @property
    def validation_set(self):
        return self.__validation_set

    @validation_set.setter
    def validation_set(self, validation_set):
        self.__validation_set = validation_set

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, test_set):
        self.__test_set = test_set


class Simple_Holdout(Validation_Technique):

    def __init__(self):
        super().__init__('Simple_holdout')

    def split(self,*args):
        if len(args)==2:
            print("HE")
            temp_array = np.split(args[0], args[1])
            self.training_set = temp_array[0]
            self.validation_set = temp_array[0]
            self.test_set = temp_array[0]
            return self.training_set, self.validation_set, self.test_set

        elif len(args)==1:
            print("HO")
            self.training_set = args[0][:int(len(args[0]) * 0.5)]  # get first 50% of file list
            self.validation_set = args[0][-int(len(args[0]) * 0.25):]  # get middle 25% of file list
            self.test_set = args[0][-int(len(args[0]) * 0.25):]  # get last 25% of file list
            return self.training_set, self.validation_set, self.test_set
        else:
            print("wrong usage of the function")



class KFold(Validation_Technique):
    def __init__(self):
        super().__init__('KFold Technique')

    def split(self, data, splits=5):
        temp_array = np.split(data, splits)
        for x in temp_array:
            tmp_test = []
            self.validation_set.append(x)
            for y in temp_array:
                if not np.array_equal(x, y):
                    tmp_test.append(y)
            self.training_set.append(tmp_test)
        for index in range(len(self.training_set)):
            print("Training_set")
            print(self.training_set[index])
            print("Val_set")
            print(self.validation_set[index])
        return self.training_set, self.validation_set


class Double_cross_validation(Validation_Technique):
    def __init__(self):
        super().__init__('Double cross validation')
        self.__outer_holdout_splitter = Simple_Holdout
        self.__inner_kFold_splitter = KFold

    @property
    def outer_houldout_splitter(self):
        return self.__outer_houldout_splitter

    @outer_houldout_splitter.setter
    def outer_houldout_splitter(self, outer_houldout_splitter):
        self.__outer_houldout_splitter = outer_houldout_splitter

    @property
    def inner_kFold_splitter(self):
        return self.__inner_kFold_splitter

    @inner_kFold_splitter.setter
    def inner_kFold_splitter(self, inner_kFold_splitter):
        self.__inner_kFold_splitter = inner_kFold_splitter

    def split(self, data, splits=5):
        tmp_test, tmp_val, tmp_test = self.__outer_houldout_splitter(data)
