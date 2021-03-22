import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        #print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')
        return (self.df)

    def prepare_X(self, df):
        cpclass2 = CarPrice()
        self.df = cpclass2.trim()
        #print(self.df)
        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        df_num = self.df[base]
        #print(df_num)
        df_num = df_num.fillna(0)
        X = df_num.values
        #print(X)
        return X

    def validate(self):
        cpclass = CarPrice()
        self.df = cpclass.trim()
        np.random.seed(2)

        n = len(self.df)
        #print(n)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()

        y_train_orig = df_train.msrp.values
        y_val_orig = df_val.msrp.values
        y_test_orig = df_test.msrp.values

        y_train = np.log1p(df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        y_test = np.log1p(df_test.msrp.values)

        del df_train['msrp']
        #del df_val['msrp']
        del df_test['msrp']

        df = df_train

        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        df_num = df[base]
        df_num = df_num.fillna(0)
        X_train = df_num.values

        #print(df_train)
        #X_train = cpclass.prepare_X(df_train)
        #print(X_train)
        #print(y_train)
        w_0, w = cpclass.linear_regression(X_train,y_train)

        y_pred = w_0 + X_train.dot(w)

        resultsdf = pd.DataFrame()
        predictedmsrp = pd.DataFrame()
        msrppreddict = {}

        for i in range(0,5):
            #i = 0
            ad = df_val.iloc[i].to_dict()
            resultsdf = resultsdf.append(ad, ignore_index=True)


            base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
            df_num = pd.DataFrame([ad])[base]
            df_num = df_num.fillna(0)
            X_test = df_num.values
            X_test = X_test[0]
            y_pred = w_0 + X_test.dot(w)

            msrppreddict['msrp_pred'] = np.expm1(y_pred)
            predictedmsrp = predictedmsrp.append(msrppreddict, ignore_index=True)
            resultsdf['msrp_pred'] = predictedmsrp
      
        print(resultsdf[['engine_cylinders', 'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity', 'msrp', 'msrp_pred']])

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        #print(X.T)
        #print(y)

        w = XTX_inv.dot(X.T).dot(y)

        
        return w[0], w[1:]
        

if __name__ == "__main__":
    cp = CarPrice()
    #cp.trim()
    cp.validate()