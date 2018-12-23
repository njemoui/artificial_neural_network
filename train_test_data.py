from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Data:
    def __init__(self,path):


        self.observations , self.labels, self.stander_scaler = None, None, None

        self.load_data(path,3,13)

        self.encoding_categorical_variables()

        self.observations_train, self.observations_test, self.labels_train, self.labels_test = train_test_split(self.observations, self.labels, test_size=0.2, random_state=0)

        self.feature_scaling()





    def load_data(self,file_path,observations_start,labels):

        dataset = pd.read_csv(file_path)

        self.observations, self.labels = dataset.iloc[:, observations_start:labels].values, dataset.iloc[:, labels].values

    def encoding_categorical_variables(self):
        labelencoder_x_1 = LabelEncoder()
        self.observations[:, 1] = labelencoder_x_1.fit_transform(self.observations[:, 1])
        labelencoder_x_2 = LabelEncoder()
        self.observations[:, 2] = labelencoder_x_2.fit_transform(self.observations[:, 2])
        one_hot_encoder = OneHotEncoder(categorical_features=[1])
        self.observations = one_hot_encoder.fit_transform(self.observations).toarray()
        self.observations = self.observations[:, 1:]


    def feature_scaling(self):

        stander_scaler = StandardScaler()

        self.observations_train, self.observations_test, self.stander_scaler = stander_scaler.fit_transform(self.observations_train), stander_scaler.transform(self.observations_test), stander_scaler

