import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

class Algo:
    def __init__(self):
        self.croissant = pd.read_csv("Croissant_Sales.csv", header = None)
        self.coffee = pd.read_csv("Coffee_Sales.csv", header = None)
        self.croiSales = self.croissant.iloc[:,1]
        self.cappySales = self.coffee.iloc[:,1]
        self.ameriSales = self.coffee.iloc[:,2]
        self.croiDates = self.croissant.iloc[:,0]
        self.coffDates = self.coffee.iloc[:,0]

        self.croi = pd.DataFrame({
            'date': pd.to_datetime(self.croiDates, format = '%d/%m/%Y'),       
            'value':  self.croiSales,   
            'product': 'croi'
            })
        
        self.cappy = pd.DataFrame({
            'date': pd.to_datetime(self.coffDates, format = '%d/%m/%Y'),        
            'value':  self.cappySales,    
            'product': 'cappy'
            })
        self.ameri = pd.DataFrame({
            'date': pd.to_datetime(self.coffDates, format = '%d/%m/%Y'),       
            'value':  self.ameriSales,   
            'product': 'ameri'
            })
        self.all_products = pd.concat([self.croi, self.cappy, self.ameri], 
                               ignore_index=True)
        self.all_products['product_encoded'] = self.all_products['product'].astype('category').cat.codes

        
        self.features = ['day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekend', 'lag_1', 'lag_7', 'lag_14', 'lag_30',
            'rolling_7', 'rolling_14', 'rolling_30']
        self.products = ['self.croi']

    def featureCreation(self, df):
        df = df.copy()
        df['day_of_week']  = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month']        = df['date'].dt.month
        df['quarter']      = df['date'].dt.quarter      
        df['is_weekend']   = df['day_of_week'].isin([5, 6]).astype(int)

        df = df.sort_values(['product', 'date'])
        df['lag_1']  = df.groupby('product')['value'].shift(1)
        df['lag_7']  = df.groupby('product')['value'].shift(7)
        df['lag_14'] = df.groupby('product')['value'].shift(14) 
        df['lag_30'] = df.groupby('product')['value'].shift(30)

        df['rolling_7']  = df.groupby('product')['value'].transform(
                               lambda x: x.rolling(7).mean())
        df['rolling_14'] = df.groupby('product')['value'].transform(
                               lambda x: x.rolling(14).mean())  
        df['rolling_30'] = df.groupby('product')['value'].transform(
                               lambda x: x.rolling(30).mean())
        return df.dropna()

    def Predictor(self):
        self.all_products = self.featureCreation(self.all_products)
        self.split     = int(len(self.all_products) * 0.8)
        self.X_train = self.all_products[self.features][:self.split]
        self.X_test    = self.all_products[self.features][self.split:]
        self.y_train = self.all_products['value'][:self.split]
        self.y_test    = self.all_products['value'][self.split:]
        model = XGBRegressor(
        n_estimators      = 1000,   # number of trees
        learning_rate     = 0.01,   # how much each tree contributes
        max_depth         = 5,      # tree complexity
        subsample         = 0.8,    # random sample of rows per tree
        colsample_bytree  = 0.8,    # random sample of features per tree
        early_stopping_roundate = 50  # stop if no improvement
        )
        model.fit(
        self.X_train, self.y_train,
        eval_set  = [(self.X_test, self.y_test)],
        verbose   = False
        )
        predictions = model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, predictions)
        print(f"MAE: {mae:.2f} units")
        return -1
    
if __name__ == "__main__":
    algo = Algo()
    algo.Predictor()