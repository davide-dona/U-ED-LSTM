import pandas as pd
import numpy as np
from functools import partial
import sklearn
import sklearn.preprocessing
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin

class CSV2EventLog:                               
    """
    Class for loading event logs in csv format, adds end-of-sequence (EOS) events,
    add new (feature engineered) columns and stores all in a pandas dataframe.
    """
   
    def __init__(self,
                 event_log_dir : str,
                 timestamp_name : str,
                 case_name : str,
                 categorical_columns : list[str],
                 continuous_columns : list[str],
                 continuous_positive_columns : list[str],
                 time_since_case_start_column : str | None = None,
                 time_since_last_event_column : str | None = None,
                 day_in_week_column : str | None = None,
                 seconds_in_day_column : str | None = None,
                 date_format : str = '%Y-%m-%d %H:%M:%S.%f',
                 min_suffix_size : int = 1,
                 **kwargs):
        """
        Creates pandas DataFrame from csv event log
        Args:
            event_log_dir (str): path to the event log
            timestamp_name (str): name of the timestamp column
            case_name (str): name of the case name columm
            categorical_columns (list[str]): names of categorical columns
            continuous_columns (list[str]): names of continuous columns
            time_since_case_start_column (str | None, optional): _description_. Defaults to None.
            time_since_last_event_column (str | None, optional): _description_. Defaults to None.
            day_in_week_column (str | None, optional) : _description_. Defaults to None.
            seconds_in_day_column (str | None, optional) : _description_. Defaults to None.
            date_format (_type_, optional): _description_. Defaults to '%Y-%m-%d %H:%M:%S.%f'.
        """
        self.case_name = case_name
        self.timestamp_name = timestamp_name
        self.time_since_case_start_column = time_since_case_start_column
        self.time_since_last_event_column = time_since_last_event_column
        self.day_in_week_column = day_in_week_column
        self.seconds_in_day_column = seconds_in_day_column
        self.date_format = date_format
        self.min_suffix_size = min_suffix_size
        
        self.df = pd.read_csv(event_log_dir)
        self.df[self.timestamp_name] = pd.to_datetime(self.df[self.timestamp_name], format=date_format, errors='coerce')

        # create new time since case started column if desired
        if self.time_since_case_start_column:
            self.__create_time_since_case_start_column()

        # create new offset time to last event column if desired
        if self.time_since_last_event_column:
            self.__create_time_since_last_event_column()

        # create new day in week column if desired
        if self.day_in_week_column:
            self.__create_day_in_week_column()

        # create new seconds in day column if desired
        if self.seconds_in_day_column:
            self.__create_seconds_in_day_column()
        
        # Add EOS to every case
        self.df = self.df.groupby(self.case_name, group_keys=False).apply(
            lambda group : self.__add_last_rows(group)).reset_index(drop=True)

        for categorical_col in categorical_columns:
            self.df[categorical_col] = self.df[categorical_col].apply(lambda x: x if pd.isna(x) else str(x))
            self.df[categorical_col] = self.df[categorical_col].astype(object)

        for continuous_col in continuous_columns:
            self.df[continuous_col] = self.df[continuous_col].astype('float32')
        for continuous_col in continuous_positive_columns:
            self.df[continuous_col] = self.df[continuous_col].astype('float32')

    def __create_time_since_case_start_column(self):
        case_start_times = self.df.groupby(self.case_name)[self.timestamp_name].transform('min')
        time_offset = self.df[self.timestamp_name] - case_start_times
        time_offset_seconds = time_offset.dt.total_seconds()
        self.df[self.time_since_case_start_column] = time_offset_seconds
        self.max_case_length = self.df.groupby(self.case_name).size().max()

    @staticmethod
    def __min_timestamp_before_event(group, timestamp_name, new_column_name):
        min_values = []
        for i, row in group.iterrows():
            before_values = group[(group[timestamp_name] < row[timestamp_name])][timestamp_name]
            if not before_values.empty:
                min_values.append(before_values.max())
            else:
                min_values.append(np.nan)
        group[new_column_name] = min_values
        return group
                                   
    def __create_time_since_last_event_column(self):
        min_timestamp_before = partial(CSV2EventLog.__min_timestamp_before_event,
                                       timestamp_name = self.timestamp_name,
                                       new_column_name = self.time_since_last_event_column)
        self.df = self.df.groupby(self.case_name).apply(min_timestamp_before).reset_index(drop=True)
        self.df[self.time_since_last_event_column] = (self.df[self.timestamp_name] - self.df[self.time_since_last_event_column]).dt.total_seconds()

    def __create_day_in_week_column(self):
        self.df[self.day_in_week_column] =  self.df[self.timestamp_name].dt.weekday

    def __create_seconds_in_day_column(self):
        self.df[self.seconds_in_day_column] = self.df[self.timestamp_name].dt.hour * 3600 + \
            self.df[self.timestamp_name].dt.minute * 60 + \
            self.df[self.timestamp_name].dt.second

    def __add_last_rows(self, group):
        new_row = {}
        for col in group.columns:
            if col == self.case_name:
                new_row[col] = group.name
            elif group[col].dtype == 'object' or group[col].dtype.name == 'category':
                new_row[col] = 'EOS'

        eos_rows = pd.DataFrame(self.min_suffix_size * [new_row])
        concat_case = pd.concat([group.sort_values(by=self.timestamp_name), eos_rows])
        return concat_case


class EventLogSplitter:
    def __init__(self,
                 train_validation_size : float,
                 test_validation_size : float,
                 **kwargs):
        """
        Splits the event log into train/test/validation event logs
        
        Args:
            train_validation_size (float): _description_
            test_validation_size (float): _description_
        """
        self.train_validation_size = train_validation_size
        self.test_validation_size = test_validation_size

    def split(self,
              event_log : CSV2EventLog):
        cases = event_log.df[event_log.case_name].unique()
        np.random.shuffle(cases)

        train_validation_ix = int(self.train_validation_size * len(cases))
        test_validation_ix = train_validation_ix + int(self.test_validation_size * len(cases))

        train_validation_cases = cases[:train_validation_ix]
        test_validation_cases = cases[train_validation_ix:test_validation_ix]
        train_cases = cases[test_validation_ix:]

        train_df = event_log.df[event_log.df[event_log.case_name].isin(train_cases)]
        train_validation_df = event_log.df[event_log.df[event_log.case_name].isin(train_validation_cases)]
        test_validation_df = event_log.df[event_log.df[event_log.case_name].isin(test_validation_cases)]

        return train_df, train_validation_df, test_validation_df

class PositiveStandardizer_normed(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        print("Positive Standardization")
        log_x = np.log1p(X)
        print("min,25%,50%,75%,max:", np.percentile(log_x, [0,25,50,75,100]))
        # Standardize values
        self.mean_ = np.mean(log_x, axis=0)
        print("Mean: ", self.mean_)
        self.std_ = np.std(log_x, axis=0)
        print("Std: ", self.std_)
        return self
        
    def transform(self, X):
        # log the observations to assume normal PDF
        log_x = np.log1p(X)
        x_enc = (log_x - self.mean_) / self.std_ 
        return x_enc
    
    def inverse_transform(self, X_enc):
        # Destandardization
        log_x = X_enc * self.std_ + self.mean_
        # Exponentiation:
        x = np.expm1(log_x)
        return x
    
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
                 continuous_positive_columns : list[str] = [],
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
        if window_size == 'auto':
            # get max. length of (100-1.5)% of the longest cases as prefix
            # and add the min. suffix_size
            case_sizes = self.event_log.groupby(case_name).size()
            self.window_size = round(case_sizes.quantile(1 - 0.015)) + self.min_suffix_size
        else:
            self.window_size = window_size
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.continuous_positive_columns = continuous_positive_columns

        self.categorical_imputers : dict[str, SimpleImputer] = dict()
        self.categorical_encoders : dict[str, sklearn.preprocessing.OrdinalEncoder]  = dict()
        for categorical_column in categorical_columns:
            self.categorical_encoders[categorical_column] = self.__get_categorical_encoder()

        self.continuous_imputers = dict()
        self.continuous_encoders : dict[str, sklearn.preprocessing.StandardScaler] = dict()
        
        # Normal encoding
        for continuous_column in continuous_columns:
            self.continuous_imputers[continuous_column] = self.__get_continuous_imputer()
            self.continuous_encoders[continuous_column] = self.__get_continuous_encoder()
        
        for continuous_positive_column in continuous_positive_columns:
            self.continuous_imputers[continuous_positive_column] = self.__get_continuous_positive_imputer()
            self.continuous_encoders[continuous_positive_column] = self.__get_continuous_positive_encoder()

    """
    Old version:
    
    def train_imputers_encoders(self):
        for col, categorical_encoder in self.categorical_encoders.items():
            column_data = np.array(self.event_log[[col]], dtype=object)
            categorical_encoder.fit(column_data)
        for col, continuous_encoder in self.continuous_encoders.items():
            continuous_imputer = self.continuous_imputers[col]
            column_data = self.event_log[[col]]
            column_data = continuous_imputer.fit_transform(column_data)
            continuous_encoder.fit(column_data)
    """
    
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
        #categorical_sizes = []
        all_categories = [[], []]
        for col in tqdm(self.categorical_columns, desc='categorical tensors'):
            if col == self.concept_name:
                case_ids, enc_column, categories, max_classes = self.encode_categorical_column(df, col, return_case_ids=True)
            else:
                enc_column, categories, max_classes = self.encode_categorical_column(df, col)
            categorical_tensors.append(enc_column)
            all_categories[0].append((col, max_classes, categories))
        continuous_tensors = []
        for col in tqdm(self.continuous_columns + self.continuous_positive_columns, desc='continouous tensors'):
            continuous_tensors.append(self.encode_continuous_column(df, col))
            all_categories[1].append((col, 1, dict()))
        return (tuple(categorical_tensors), tuple(continuous_tensors), tuple(case_ids)), tuple(all_categories)

    # Corrected verison:
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

    """
    Old version:
    
    def encode_categorical_column(self, df, col, return_case_ids=False):
        grouped = df.groupby(self.case_name)
        windows = []
        categories = {category: idx + 1 for idx, category in enumerate(self.categorical_encoders[col].categories_[0])}
        
        case_ids = []
        for case_id, group in tqdm(grouped, desc=col, leave=False):
            if return_case_ids:
                case_ids.extend([case_id] * len(group))
            case_values = np.array(group[[col]], dtype=object)
            case_values_enc = self.categorical_encoders[col].transform(case_values) + 1
            # Pad encodings
            padded_encodings = []
            for i in range(self.min_suffix_size - 1, len(case_values_enc)):
                padded_encodings.append(self.pad_to_window_size(case_values_enc[:i+1]))
            windows.extend(padded_encodings)
        
        # Convert to tensor
        padded_array = np.array(windows, dtype=int)
        t = torch.tensor(padded_array, dtype=torch.long)
        
        max_classes = len(self.categorical_encoders[col].categories_[0]) + 1
        if return_case_ids:
            return case_ids, t.squeeze(-1), categories, max_classes
        else:
            return t.squeeze(-1), categories, max_classes   
    """

    """
    Old version:
    
    def encode_continuous_column(self, df, col):
        grouped = df.groupby(self.case_name)
        windows = []
        for case_id, group in tqdm(grouped, desc=col, leave=False):
            case_values = group[[col]]
            case_values_enc = self.continuous_imputers[col].transform(case_values)
            case_values_enc = self.continuous_encoders[col].transform(case_values_enc)
            padded_encodings = []
            for i in range(self.min_suffix_size - 1, len(case_values_enc)):
                padded_encodings.append(self.pad_to_window_size(case_values_enc[:i+1]))
            windows.extend(padded_encodings)
        # Convert to tensor
        padded_array = np.array(windows)
        t = torch.tensor(padded_array, dtype=torch.float32)
        return t.squeeze(-1)
    """
    
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
    
    """
    Old version
    
    def pad_to_window_size(self, previous_values):
        if len(previous_values) > self.window_size:
            return previous_values[-self.window_size:].tolist()
        else:
            return [[0.0]] * (self.window_size - len(previous_values)) \
                   + previous_values[-self.window_size:].tolist()
    """
    
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

    def decode_event(self, event_tuple : tuple):
        cat, cont, case_id = event_tuple
        decoded_event = dict()
        for i, col in enumerate(self.categorical_columns):
            enc_col = cat[i].unsqueeze(-1).numpy()
            if col in self.categorical_encoders:
                categories = self.categorical_encoders[col].categories_[0]
                dec_col = np.array([categories[idx - 1] if idx > 0 and idx <= len(categories) else np.nan for idx in enc_col.flatten()])
            else:
                dec_col = enc_col
            decoded_event[col] = dec_col.tolist()
        for i, col in enumerate(self.continuous_columns + self.continuous_positive_columns):
            enc_col = cont[i].unsqueeze(-1).numpy()
            if col in self.continuous_encoders:
                dec_col = self.continuous_encoders[col].inverse_transform(enc_col)
            else:
                dec_col = enc_col
            decoded_event[col] = dec_col.flatten().tolist()
        decoded_event[self.case_name] = [case_id] * len(decoded_event[self.categorical_columns[0]])
        return pd.DataFrame(decoded_event)
    
    def __get_continuous_imputer(self):
        return SimpleImputer(strategy='mean')

    def __get_categorical_encoder(self):
        return sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                                    unknown_value=-1,
                                                    encoded_missing_value=-1)

    def __get_continuous_encoder(self):
        return sklearn.preprocessing.StandardScaler()

    def __get_continuous_positive_imputer(self):
        return SimpleImputer(strategy='mean')
    
    def __get_continuous_positive_encoder(self):
        standardizer = PositiveStandardizer_normed()
        return standardizer


class EventLogLoader:
    def __init__(self, event_log_location, event_log_properties, suffix_len = 1):
        self.event_log = CSV2EventLog(event_log_location, **event_log_properties)
        splitter = EventLogSplitter(**event_log_properties)
        self.train_df, self.val_df, self.test_df = splitter.split(self.event_log)

        self.encoder_decoder = TensorEncoderDecoder(self.train_df, **event_log_properties)
        
        # Data are transformed
        self.encoder_decoder.train_imputers_encoders()

    def get_dataset(self, type : str):
        if type == 'train':
            df = self.train_df
        elif type == 'val':
            df = self.val_df
        elif type == 'test':
            df = self.test_df
        encoded_data, all_categories = self.encoder_decoder.encode_df(df)
        return EventLogDataset(encoded_data, all_categories, self.encoder_decoder)

class EventLogDataset(Dataset):
    def __init__(self,
                 tensor_tuple : tuple,
                 all_categories : tuple[list[tuple[str, int, dict[str : int]]]],
                 encoder_decoder : TensorEncoderDecoder):
        self.tensor_list : tuple = tensor_tuple
        self.all_categories : tuple[list[tuple[str, int, dict[str : int]]]] = all_categories
        self.encoder_decoder : TensorEncoderDecoder = encoder_decoder

    def __len__(self):
        if len(self.tensor_list[0]):
            return self.tensor_list[0][0].shape[0]
        else:
            return self.tensor_list[1][0].shape[0]

    def __getitem__(self, idx):
        cat = list()
        #categorical items
        for i in self.tensor_list[0]:
            cat.append(i[idx])
        #continuous items
        cont = list()
        for i in self.tensor_list[1]:
            cont.append(i[idx])
        #case ids
        case_id = self.tensor_list[2][idx]
        return (tuple(cat), tuple(cont), case_id)