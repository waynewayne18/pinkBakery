import pandas as pd
import numpy as np
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
        self.all_products['product_cat'] = self.all_products['product'].astype('category').cat.codes

        
        self.features = ['day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekend', 'lag_1', 'lag_7', 'lag_14', 'lag_30',
            'rolling_7', 'rolling_14', 'rolling_30']

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

        df['rolling_7']  = df.groupby('product')['value'].transform(lambda x: x.rolling(7).mean())
        df['rolling_14'] = df.groupby('product')['value'].transform(lambda x: x.rolling(14).mean())  
        df['rolling_30'] = df.groupby('product')['value'].transform(lambda x: x.rolling(30).mean())
        return df.dropna()
    
    def forecast(self, days=28):    
        history = self.all_products.copy()
        all_forecasts = []
       
        for product in ['croi', 'cappy', 'ameri']:
            product_history = history[history['product'] == product].copy()
            product_history = product_history.sort_values('date').reset_index(drop=True)
            product_cat = product_history['product_cat'].iloc[0]
            future_predictions = []
            
            for day in range(1, days + 1):
                future_date = product_history['date'].max() + pd.Timedelta(days = 1)
                recent_values = product_history['value'].values
                row = {
                    'day_of_week':  future_date.dayofweek,
                    'day_of_month': future_date.day,
                    'month':        future_date.month,
                    'quarter':      future_date.quarter,
                    'is_weekend':   int(future_date.dayofweek in [5, 6]),
                    #lag makes features from last x days more important
                    'lag_1':  recent_values[-1],
                    'lag_7':  recent_values[-7],
                    'lag_14': recent_values[-14],
                    'lag_30': recent_values[-30],
 
                    'rolling_7':  np.mean(recent_values[-7:]),
                    'rolling_14': np.mean(recent_values[-14:]),
                    'rolling_30': np.mean(recent_values[-30:]),
                }
                # prediction                 
                future = pd.DataFrame([row])[self.features]
                pred = self.models[product].predict(future)[0]
                
                translator = {'croi': 'Croissant', 'cappy': 'Cappuccino', 'ameri': 'Americano'}
                future_predictions.append({
                    'date':    future_date,
                    'product': translator[product],
                    'forecast': round(pred, 0)
                })
                # append prediction to history so next day can use it as lag
                new_row = pd.DataFrame([{
                    'date':    future_date,
                    'value':   pred,
                    'product': product,
                    'product_cat': product_cat
                }])
                product_history = pd.concat(
                    [product_history, new_row], 
                    ignore_index=True
                )
            
            all_forecasts.extend(future_predictions)
        return pd.DataFrame(all_forecasts)

    def Predictor(self):
        maes = []
        self.all_products = self.featureCreation(self.all_products)
        self.models = {}  # dictionary to store one model per product
        for product in ['croi', 'cappy', 'ameri']:
            # filter data for this product only
            product_data = self.all_products[
                self.all_products['product'] == product
            ]
            split = int(len(product_data) * 0.8)
            
            training1 = product_data[self.features][:split]
            testing1  = product_data[self.features][split:]
            training2 = product_data['value'][:split]
            testing2  = product_data['value'][split:]
            
            model = XGBRegressor(
                n_estimators      = 1000,
                learning_rate     = 0.01,
                max_depth         = 5,
                subsample         = 0.8,
                colsample_bytree  = 0.8,
                early_stopping_rounds = 50
            )
            model.fit(
                training1, training2,
                eval_set = [(testing1, testing2)],
                verbose  = False
            )
            predictions = model.predict(testing1)
            mae = mean_absolute_error(testing2, predictions)

            translator = {'croi': 'Croissant', 'cappy': 'Cappuccino', 'ameri': 'Americano'}
            print(f"{translator[product]} MAE: {mae:.2f} units")
            maes.append(mae)
            # store model under product key
            self.models[product] = model
        return maes
    
if __name__ == "__main__":
    algo = Algo()
    algo.Predictor()
    forecast_df = algo.forecast(days=28)
    print(forecast_df)
