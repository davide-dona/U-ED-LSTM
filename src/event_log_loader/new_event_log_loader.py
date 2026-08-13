import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.impute import SimpleImputer
import torch
from tqdm.notebook import tqdm

class TensorEncoderDecoder:
    """
    Class for encoding event log data (pandas dataframe)
    into a torch tensor data structure and decoding it
    back to a dataframe

    We store all attributes as individual tensors
    """

    def __init__(self,
                 event_log : pd.DataFrame,
                 case_name : str,
                 concept_name : str,
                 window_size : int,
                 min_suffix_size : int,
                 categorical_columns : list[str] = [],
                 continuous_columns : list[str] = [],
                 **kwargs):
        """_summary_

        Args:
            event_log (CSV2EventLog): _description_
            case_name (str): _description_
            window_size (int | str): either absolute window size, or 'auto'.
                                     Then top 1.5% of case length is taken.
            min_suffix_size (int) : Min number of suffix events, i.e., number of EOS events added to the case
            categorical_columns (list[str], optional): _description_. Defaults to [].
            continuous_columns (list[str], optional): _description_. Defaults to [].
        """
        self.event_log = event_log
        self.case_name = case_name
        self.concept_name = concept_name
        self.min_suffix_size = min_suffix_size
        self.window_size = window_size
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        
        # Define imputers and encoders for each categorical column
        self.categorical_encoders : dict[str, sklearn.preprocessing.OrdinalEncoder]  = dict()
        for categorical_column in categorical_columns:
            self.categorical_encoders[categorical_column] = self.__get_categorical_encoder()
        
        # Define imputers and encoders for each continuous column
        self.continuous_imputers = dict()
        self.continuous_encoders : dict[str, sklearn.preprocessing.StandardScaler] = dict()
        for continuous_column in continuous_columns:
            self.continuous_imputers[continuous_column] = self.__get_continuous_imputer()
            self.continuous_encoders[continuous_column] = self.__get_continuous_encoder()


    def train_imputers_encoders(self):
        # categorical encoders: fit on 2D numpy arrays with dtype=object
        for col, categorical_encoder in self.categorical_encoders.items():
            column_data = self.event_log[[col]].astype(object).to_numpy()  # shape (n,1)
            categorical_encoder.fit(column_data)

        # continuous encoders / imputers: fit on 2D numpy arrays (n_samples, 1)
        for col, continuous_encoder in self.continuous_encoders.items():
            continuous_imputer = self.continuous_imputers[col]
            column_data = self.event_log[[col]].to_numpy()  # DataFrame -> ndarray (n,1)
            column_data = continuous_imputer.fit_transform(column_data)  # still (n,1)
            continuous_encoder.fit(column_data)  # StandardScaler or custom transformer expects 2D

    def encode_df(self, df) -> tuple[tuple[torch.Tensor, torch.Tensor, tuple],
                                     tuple[list[tuple[str, int, dict[str : int]]]]]:
        categorical_tensors = []
        all_categories = [[], []]
        for col in tqdm(self.categorical_columns, desc='categorical tensors'):
            if col == self.concept_name:
                case_ids, enc_column, categories, max_classes = self.encode_categorical_column(df, col, return_case_ids=True)
            else:
                enc_column, categories, max_classes = self.encode_categorical_column(df, col)
            categorical_tensors.append(enc_column)
            all_categories[0].append((col, max_classes, categories))
        continuous_tensors = []
        for col in tqdm(self.continuous_columns, desc='continouous tensors'):
            continuous_tensors.append(self.encode_continuous_column(df, col))
            all_categories[1].append((col, 1, dict()))
        return (tuple(categorical_tensors), tuple(continuous_tensors), tuple(case_ids)), tuple(all_categories)


    def encode_categorical_column(self, df, col, return_case_ids=False):
        grouped = df.groupby(self.case_name)
        windows = []
        categories = {category: idx + 1 for idx, category in enumerate(self.categorical_encoders[col].categories_[0])}
        
        case_ids = []
        for case_id, group in tqdm(grouped, desc=col, leave=False):
            case_values = np.array(group[[col]], dtype=object)
            case_values_enc = self.categorical_encoders[col].transform(case_values) + 1  # shape (n,1)
            # Pad encodings - clearer prefix loop (prefix_len from min_suffix_size .. len)
            padded_encodings = []
            for prefix_len in range(self.min_suffix_size, len(case_values_enc) + 1):
                padded_encodings.append(self.pad_to_window_size(case_values_enc[:prefix_len]))
            windows.extend(padded_encodings)
            if return_case_ids:
                # append one case id per generated window (not per original row)
                case_ids.extend([case_id] * len(padded_encodings))

        if len(windows) == 0:
            # avoid creating empty numpy array with ambiguous dtype
            padded_array = np.zeros((0, self.window_size), dtype=int)
        else:
            padded_array = np.array(windows, dtype=int)
        t = torch.tensor(padded_array, dtype=torch.long)

        max_classes = len(self.categorical_encoders[col].categories_[0]) + 1
        if return_case_ids:
            return case_ids, t.squeeze(-1), categories, max_classes
        else:
            return t.squeeze(-1), categories, max_classes

    
    def encode_continuous_column(self, df, col):
        grouped = df.groupby(self.case_name)
        windows = []
        for case_id, group in tqdm(grouped, desc=col, leave=False):
            case_values = group[[col]].values  # shape (n,1)
            case_values_imputed = self.continuous_imputers[col].transform(case_values)
            case_values_enc = self.continuous_encoders[col].transform(case_values_imputed)
            padded_encodings = []
            for prefix_len in range(self.min_suffix_size, len(case_values_enc) + 1):
                padded_encodings.append(self.pad_to_window_size(case_values_enc[:prefix_len]))
            windows.extend(padded_encodings)
        if len(windows) == 0:
            padded_array = np.zeros((0, self.window_size), dtype=float)
        else:
            padded_array = np.array(windows, dtype=float)
        t = torch.tensor(padded_array, dtype=torch.float32)
        return t.squeeze(-1)
    
    
    def pad_to_window_size(self, previous_values):
        """
        previous_values: array-like with shape (k, 1)
        returns list of shape (window_size, 1)
        """
        prev_list = np.asarray(previous_values).tolist()
        if len(prev_list) > self.window_size:
            return prev_list[-self.window_size:]
        else:
            pad_count = self.window_size - len(prev_list)
            # use 0.0 for continuous; for categorical it will be cast to int later when dtype=int
            return [[0.0]] * pad_count + prev_list

    def __get_continuous_imputer(self):
        return SimpleImputer(strategy='mean')

    def __get_categorical_encoder(self):
        return sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                                    unknown_value=-1,
                                                    encoded_missing_value=-1)

    def __get_continuous_encoder(self):
        return sklearn.preprocessing.StandardScaler()
