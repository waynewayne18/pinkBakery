from importlib import reload

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import random
import csv
import math




class CoffeeData:

    def run(self, filename):
        self.fileSetup(filename)
        results = self.main()
        return results

    def main(self):
    # Load and preprocess coffee data
        self.day3Avg()
        self.day7Avg()
        self.day30Avg()

        self.bias1 = self.bias1Prep()
        dailyAim7 = self.firstLayerPrep()
        dailyAim3 = self.secondLayerPrep()
        dailyAim30 = self.thirdLayerPrep()
        daySums = self.dayLayerprep()
        ml = MLComparisonWorkflow(self)
        ml.preprocess()
        ml.train_both_knn(n_neighbors=3)
        dates = [datetime.strptime(d, "%d/%m/%Y") for d in self.dates]

        # full feature columns
        X_full = np.array(self.data_x).reshape(-1, 1)
        Y_full = np.array(self.data_y).reshape(-1, 1)


        # predict y from x
        knn_y = ml.stored_models["KNN_y_on_x"]["model"]
        scaler_x_for_y = ml.stored_models["KNN_y_on_x"]["scaler"]
        X_full_scaled = scaler_x_for_y.transform(X_full)
        pred_y_from_x = knn_y.predict(X_full_scaled)

        # predict x from y
        knn_x = ml.stored_models["KNN_x_on_y"]["model"]
        scaler_y_for_x = ml.stored_models["KNN_x_on_y"]["scaler"]
        Y_full_scaled = scaler_y_for_x.transform(Y_full)
        pred_x_from_y = knn_x.predict(Y_full_scaled)

        def numpy_metrics(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")
            return {"mae": mae, "rmse": rmse, "r2": r2}

        metrics_y = numpy_metrics(Y_full.ravel(), pred_y_from_x)
        metrics_x = numpy_metrics(X_full.ravel(), pred_x_from_y)

        plt.figure(figsize=(13,5))
        plt.plot(dates, Y_full.ravel(), label="Actual Americano (data_y)", color="black", linewidth=2)
        plt.plot(dates, X_full.ravel(), label="Actual Cappuccino (data_x)", color="gray", linewidth=1, alpha=0.7)
        plt.plot(dates, pred_y_from_x, label=f"KNN predict y from x (R2={metrics_y['r2']:.3f})", color="tab:blue", linestyle="--")
        plt.plot(dates, pred_x_from_y, label=f"KNN predict x from y (R2={metrics_x['r2']:.3f})", color="tab:green", linestyle=":")
        plt.xlabel("Date")
        plt.ylabel("Count sold")
        plt.title("Actual series and two KNN predictions")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()

        return {
            "bias1": self.bias1,
            "dailyAim7": dailyAim7,
            "dailyAim3": dailyAim3,
            "dailyAim30": dailyAim30,
            "daySums": daySums,
        }

    def __init__(self, datafilename: str):
            self.dates = []
            self.data_x = []
            self.data_y = []

            self.c3Avg = []
            self.a3Avg = []
            self.c7Avg = []
            self.a7Avg = []
            self.c30Avg = []
            self.a30Avg = []
            self.randNullLst = []
            self.randNullLst0 = []
            self.W1 = 0.5
            self.W2 = 0.3
            self.A1 = 0
            self.A2 = 0
            self.Z1 = 0
            self.Z2 = 0
            self.bias1 = 0
            self.bias2 = 0

            self.fileSetup(datafilename)

    def fileSetup(self, filename):
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            next(reader)
            for row in reader:
                self.dates.append(row[0])
                self.data_x.append(int(row[1]))
                self.data_y.append(int(row[2]))
    
    def day3Avg(self):
        if len(self.dates) % 3 != 0:
            a = (len(self.dates) % 3)
            d3Avg = [0] * ((len(self.dates)-(a)) // 3)
        else:
            d3Avg = ([0] * (len(self.dates) // 3))
        for e in range (len(self.dates)//3):
            d3Avg[e] = (e*3)
        for i in range(0, len(self.dates) - len(self.dates) % 3, 3):
            c = sum(self.data_x[i:i+3]) / 3
            self.c3Avg.append(c)
            d = sum(self.data_y[i:i+3]) / 3
            self.a3Avg.append(d)
        if len(self.dates) % 3 != 0:
            f = (len(self.dates) % 3)
            c = sum(self.data_x[-f:])
            a = sum(self.data_y[-f:])
            c = c / f
            a = a / f
            self.c3Avg.append(c)
            self.a3Avg.append(a)
        return d3Avg, self.c3Avg, self.a3Avg

    def day7Avg(self):
        if len(self.dates) % 7 != 0:
            a = (len(self.dates) % 7)
            d7Avg = [0] * ((len(self.dates)-(a)) // 7)
        else:
            d7Avg = ([0] * (len(self.dates) // 7))
        for e in range (len(self.dates)//7):
            d7Avg[e] = (e*7)
        for i in range(0, len(self.dates) - len(self.dates) % 7, 7):
            c = sum(self.data_x[i:i+7]) / 7
            self.c7Avg.append(c)
            d = sum(self.data_y[i:i+7]) / 7
            self.a7Avg.append(d)
        if len(self.dates) % 7 != 0:
            f = (len(self.dates) % 7)
            c = sum(self.data_x[-f:])
            a = sum(self.data_y[-f:])
            c = c / f
            a = a / f
            self.c7Avg.append(c)
            self.a7Avg.append(a)
        return d7Avg, self.c7Avg, self.a7Avg


    def day30Avg(self):
        if len(self.dates) % 30 != 0:
            a = (len(self.dates) % 30)
            self.d30Avg = [0] * ((len(self.dates)-(a)) // 30)
        else:
            self.d30Avg = ([0] * (len(self.dates) // 30))
        for e in range (len(self.dates)//30):
            self.d30Avg[e] = (e)
        for i in range(0, len(self.dates) - len(self.dates) % 30, 30):
            c = sum(self.data_x[i:i+30]) / 30
            self.c30Avg.append(c)
            d = sum(self.data_y[i:i+30]) / 30
            self.a30Avg.append(d)
        if len(self.dates) % 30 != 0:
            f = (len(self.dates) % 30)
            c = sum(self.data_x[-f:])
            a = sum(self.data_y[-f:])
            c = c / f
            a = a / f
            self.c30Avg.append(c)
            self.a30Avg.append(a)
        return self.d30Avg, self.c30Avg, self.a30Avg

    def bias1Prep(self):
        if len(self.data_y) == 0:
            return 0.0
        for j in range(100):
            for i in range(10):
                r = random.randint(0, len(self.data_y)-1)
                self.randNullLst.append(self.data_y[r])
            self.bias1 = sum(self.randNullLst)/len(self.randNullLst)
        for i in range(20):
            self.randNullLst0.append(random.randint(0, len(self.data_y)-1))
            for i in range(len(self.randNullLst0)):
                temp = abs(self.bias1/self.randNullLst0[i])
                self.randNullLst0[i] = temp
        self.bias1 += sum(self.randNullLst0)/len(self.randNullLst0)
        self.bias1 = self.bias1/2
        return self.bias1


    def firstLayerPrep(self):
        gradients = []
        for i in range(len(self.c7Avg) - 1):
            gradient = (self.c7Avg[i+1] - self.c7Avg[i]) / 7
            gradients.append(gradient)
        if len(gradients) == 0:
            return 0.0
        dailyAim = sum(gradients) / len(gradients)
        return dailyAim

# Apply the same pattern to secondLayerPrep and thirdLayerPrep,
# changing the divisor (3 or 30) as you already have.

    def secondLayerPrep(self):    # Using 3 day Avg to find the gradient in sales at each week. 
        gradients = []
        for i in range (len(self.c3Avg)-1):
            gradient = (self.c3Avg[i+1] - self.c3Avg[i])/3
            gradients.append(gradient)
        if len(gradients) == 0:
            return 0.0
        dailyAim = sum(gradients)/len(gradients)
        return dailyAim

    def thirdLayerPrep(self):    # Using 30 day Avg to find the gradient in sales at each week. 
        gradients = []
        for i in range (len(self.c30Avg)-1):
            gradient = (self.c30Avg[i+1] - self.c30Avg[i])/30
            gradients.append(gradient)
        if len(gradients) == 0:
            return 0.0  
        dailyAim = sum(gradients)/len(gradients)
        return dailyAim

    def dayLayerprep(self):
        daySum = []
        totalSums = []
        weeks = (len(self.dates)) // 7
        for j in range(7):
            for i in range(weeks):
                idx = (7 * i) + j
                if idx < len(self.data_y):
                    daySum.append(self.data_y[idx])
            if len(daySum) == 0:
                totalSums.append(0)   # or append(None) / float('nan')
            else:
                totalSums.append(sum(daySum) // len(daySum))
            daySum = []
        return totalSums


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class MLComparisonWorkflow:
    def __init__(self, coffee):
        # raw arrays
        self.x = np.array(coffee.data_x).reshape(-1, 1)   # feature for y-on-x
        self.y = np.array(coffee.data_y).reshape(-1, 1)   # feature for x-on-y (and target for y)
        self.stored_models = {}

    def _fit_and_eval(self, X, y, model, test_size=0.3, random_state=12345):
        # split, scale X only (targets left as-is)
        X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=test_size, random_state=random_state)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        return {"model": model, "scaler": scaler, "r2": r2, "mae": mae}

    def train_knn_y_on_x(self, n_neighbors=3):
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        res = self._fit_and_eval(self.x, self.y, knn)
        self.stored_models["KNN_y_on_x"] = res

    def train_knn_x_on_y(self, n_neighbors=3):
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        res = self._fit_and_eval(self.y, self.x, knn)  # swap roles
        self.stored_models["KNN_x_on_y"] = res

    # convenience: train both
    def train_both_knn(self, n_neighbors=3):
        self.train_knn_y_on_x(n_neighbors=n_neighbors)
        self.train_knn_x_on_y(n_neighbors=n_neighbors)

    def preprocess(self, test_size=0.3, random_state=12345):
        """
        Split and scale the workflow's own self.x and self.y.
        Stores the scaler and splits on the instance and returns them.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y.ravel(), test_size=test_size, random_state=random_state
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # store for later use
        self.splits = {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
        }
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler



    


def sigmoid(x):
    return 1 / (1+math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#def forwardHidden1(self):
#    self.A1=((self.a7Avg+self.a3Avg+self.a30Avg)*(max(self.data_x)/min(self.data_x)*self.W1))
#    self.Z1=sigmoid(self.A1)
#    self.A2=(self.A1*self.W2)
#    self.Z2=sigmoid(self.A2)
#
#    print(self.A1, self.A2, self.Z1, self.Z2)

def make_xor_reliability_plot(train_X, train_y):
    """

    Parameters:
    -----------
    train_f: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object
    
    ax: matplotlib.axes.Axes
        axis
    """
    
    hidden_layer_width = list(range(1, 11))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) #create a figure with 2 subplots, set fig size
    successes = np.zeros(10)
    epochs = (np.zeros((10,10)))
    efficiency = np.zeros(10)
    for i, h_nodes in enumerate(hidden_layer_width): # loop tracking i and h_nodes
        total_epochs = 0
        successful_runs = 0
        for repetition in range(10):
            num_hidden_nodes = h_nodes
            xor = MLPClassifier(
                hidden_layer_sizes=(int(h_nodes),), #Hidden layer width
                max_iter=1000,                      #Maximum itterations
                alpha=1e-4,                         #L2 regularization parameter
                solver="sgd",                       #Stochastic Gradient Descent
                #verbose=0, #print progress msg
                learning_rate_init=0.1,             
                random_state=repetition             #Seed for reproduction rate
                )
            _ = xor.fit(train_X, train_y)
            training_accuracy = 100 * xor.score(train_X, train_y)   #As a percentage
            print(f"Training set accuracy: {training_accuracy}% after {xor.n_iter_} iterations")
            if training_accuracy == 100:
                successes[h_nodes - 1] += 1
                successful_runs += 1
                total_epochs += xor.n_iter_
    if np.all(successes == 0):
        efficiency[i] = 1000
    else:
        efficiency[i] = total_epochs / successful_runs
    ax1.plot(hidden_layer_width, successes, marker='o')
    ax1.set_ylim((0.0, 10.0))
    plt.title("Reliability")
    ax1.set_xlabel("Hidden Layer Width")
    ax1.set_ylabel("Sucess Rate")
    ax2.plot(hidden_layer_width, efficiency, marker='v')
    ax2.set_ylim((0.0, 100.0))
    plt.title("Efficiency")
    ax2.set_xlabel("Hidden Layer Width")
    ax2.set_ylabel("Mean Epochs")    
    
    return fig, ax1, ax2
#(((sigmoid(self.a7Avg))*Max(sales_americano))*self.a3Avg)/self.a30Avg

coffee = CoffeeData("Coffee_Sales.csv")
coffee.run("Coffee_Sales.csv")
