class IncorrectSizeException(Exception):

    def __init__(self, message = ""):
        self.message = message

class LayerMissingException(Exception):

    def __init__(self, message = ""):
        self.message = message

class ModelIsNotReadyException(Exception):

    def __init__(self, message = ""):
        self.message = message