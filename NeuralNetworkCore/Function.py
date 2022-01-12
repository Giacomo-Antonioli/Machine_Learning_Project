class Function:
    """
    Class representing a function
    Attributes:
        function (function): Represents the function itself
        name (string): name of the function
    """

    def __init__(self, name, function):
        self.__name = name
        self.__function = function

    @property
    def name(self):
        return self.__name

    @property
    def function(self):
        return self.__function


class DerivableFunction(Function):
    """
    Class representing a function that we need the derivative of.
    Subclass of Function.
    Attributes:
        derive (function ): Represents the derivative of the function
    """

    def __init__(self, function, derive, name):
        super(DerivableFunction, self).__init__(name=name, function=function)
        self.__derive = derive

    @property
    def derive(self):
        return self.__derive
