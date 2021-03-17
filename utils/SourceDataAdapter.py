import pandas as pd
import numpy as np
import json

class SourceDataAdapter:

    def __init__(self):
        pass

    def _is_dataframe(self, data):
        if isinstance(data, pd.DataFrame):
            return True
        else:
            return False

    def _is_series(self, data):
        if isinstance(data, pd.Series):
            return True
        else:
            return False

    def _is_numpy(self, data):
        if type(data).__module__ == np.__name__:
            return True
        else:
            return False

    def _is_list(self, data):
        if isinstance(data, list):
            return True
        else:
            return False

    def _numpy_to_dataframe(self, data):
        num_columns = 0
        try:
            num_columns = data.shape[1]
        except:
            adapted_data = pd.Series(data)

            return adapted_data
            
        num_columns = data.shape[1]

        if num_columns > 1:
            adapted_data = pd.DataFrame(data, columns=[str(i) for i in range(num_columns)])     
        else:
            raise Exception("Target data is empty so it can't be transformed")

        return adapted_data
    
    def adapt_source_data(self, data):
        adapted_data = None
        if self._is_numpy(data):
            adapted_data = self._numpy_to_dataframe(data)

        elif self._is_list(data):
            data = np.array(data)
            adapted_data = self._numpy_to_dataframe(data)
            
        elif self._is_dataframe(data) or self._is_series(data):
            adapted_data = data

        else:
            raise Exception("Unable to transform data to dataframe. Please check your data formats are numpy array, list or pandas dataframe")
        
        return adapted_data