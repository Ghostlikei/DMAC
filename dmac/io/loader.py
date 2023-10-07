import json

class Loader:
    def __init__(self):
        pass

    def load_data(self, data_path):
        raise NotImplementedError

    def preprocess_data(self, data):
        raise NotImplementedError

    def get_data_loader(self, data_path):
        """
        A common method to load and preprocess data, returning a data loader.
        """
        data = self.load_data(data_path)
        preprocessed_data = self.preprocess_data(data)

        # Depending on the derived class and its specific data source,
        # you can create and return an appropriate data loader here.
        # For example, using PyTorch DataLoader or other data loading utilities.

        # Example (using PyTorch DataLoader):
        # from torch.utils.data import DataLoader, TensorDataset
        # dataset = TensorDataset(*preprocessed_data)
        # return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return preprocessed_data
    
class Project1Loader(Loader):
    def __init__(self):
        pass

    def load_data(self, data_path, logging=True):
        # Load the data
        data = []
        with open(data_path, 'r') as file:
            for line in file:
                item = json.loads(line)
                data.append(item)

        if logging:
            print("Load complete, Data[0]: ")
            print(data[0])
        
        return data

    def load(self, data_path):
        return self.load_data(data_path)
        

##### Unused code ######
# Example of a derived class for PyTorch data loading
class PyTorchLoader(Loader):
    def load_data(self, data_path):
        # Implement data loading logic specific to PyTorch
        pass

    def preprocess_data(self, data):
        # Implement data preprocessing specific to PyTorch
        pass

# Example of a derived class for scikit-learn data loading
class ScikitLearnLoader(Loader):
    def load_data(self, data_path):
        # Implement data loading logic specific to scikit-learn
        pass

    def preprocess_data(self, data):
        # Implement data preprocessing specific to scikit-learn
        pass

##### Unused code ######
