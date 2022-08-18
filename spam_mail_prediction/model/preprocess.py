import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Preprocess:
    def __init__(self, data_path):
        self.raw_data = pd.read_csv(data_path)
        self.raw_data_clean = self.raw_data.where((pd.notnull(self.raw_data)), '')
        self.label_encod()

    def label_encod(self):
        self.raw_data_clean.loc[self.raw_data_clean['Category'] == 'spam', 'Category'] = 0
        self.raw_data_clean.loc[self.raw_data_clean['Category'] == 'ham', 'Category'] = 1
        print()

    def split_test_train_data(self):
        x_inpout = self.raw_data_clean['Message']
        y_output = self.raw_data_clean['Category']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(x_inpout, y_output, test_size=0.2,
                                                                                random_state=3)

    def feature_extraction(self):
        extrac_func = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')
        self.X_train_features = extrac_func.fit_transform(self.X_train)
        self.X_test_features = extrac_func.transform(self.X_test)


worker = Preprocess('dataset/mail_data.csv')
worker.label_encod()
