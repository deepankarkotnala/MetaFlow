from metaflow import FlowSpec, step, IncludeFile, Parameter
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits
from sklearn.model_selection import train_test_split
from sklearn import ensemble

cwd = os.getcwd()
plot_path = cwd +'/plots'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class MetaFlowDemo(FlowSpec):
    """
    A flow to analyze the housing dataset and explain 
    the different factors which affect the house prices.

    The flow performs the following steps:
    1) Take input from the user - Number of Bedrooms, 
       Number of Bathrooms, and Living Area in Square Ft.
    2) Ingests a CSV file containing data about housing prices.
    3) Plots 4 plots using parallel processes (in 4 different nodes).
    4) In parallel branches:
       - A) Plot for different factors on which the price of house depends.
       - B) Calls the join node to move towards the end.
    5) Predict the House Price

    """
    df = pd.read_csv('kc_house_data.csv')
    df = df.round()
    bedrooms = Parameter('bedrooms',
                      help="Number of Bedrooms in the House",
                      default=3)
    bathrooms = Parameter('bathrooms',
                      help="Number of Bathrooms in the House",
                      default=2)
    sqft_living = Parameter('sqft_living',
                      help="Living Area in Square Ft",
                      default=2410)

    @step
    def start(self):
        """
        Call the different parallel processes for plotting graphs all at once.

        """
        print('Calling 4 Parallel Nodes to generate 4 Plots on the data')
        print('The plot preview window will automatically close in 5 seconds.')
        print('Plots will be saved in {}/plots directory'.format(cwd))
        self.next(self.node_1, self.node_2, self.node_3, self.node_4)

    @step
    def node_1(self):
        """
        Plot 1: Relation of House Price and Square Ft Area

        """
        try:
            data = self.df
            plt.figure(figsize=(8,5))
            plt.scatter(data.price,data.sqft_living)
            plt.title("Price vs Square Feet")
            plt.savefig(plot_path+'/price_vs_sq_ft.png')
            print('Plotting Graph 1')
            plt.pause(5)
        except:
            pass

        self.next(self.join)

    @step
    def node_2(self):
        """
        Plot 2: Distribution of Number of Bedrooms in the houses present in the Area

        """
        try:
            time.sleep(1)
            data = self.df
            plt.figure(figsize=(8,5))
            data['bedrooms'].value_counts().plot(kind='bar')
            plt.title('Number of Bedrooms')
            plt.xlabel('Bedrooms')
            plt.ylabel('Count')
            plt.savefig(plot_path+'/num_bedrooms.png')
            print('Plotting Graph 2')
            plt.pause(5)
        except:
            pass
        self.next(self.join)

    @step
    def node_3(self):
        """
        Plot 3: Relation between the Number of Bedrooms and House Prices

        """
        try:
            time.sleep(2)
            data = self.df
            plt.figure(figsize=(8,5))
            plt.scatter(data.bedrooms,data.price)
            plt.title("Bedroom and Price ")
            plt.xlabel("Bedrooms")
            plt.ylabel("Price")
            plt.savefig(plot_path+'/house_price_by_bedrooms.png')
            print('Plotting Graph 3')
            plt.pause(5)
        except:
            pass

        self.next(self.join)

    @step
    def node_4(self):
        """
        Plot 4: Price Distribution across ZipCodes

        """
        try:
            time.sleep(3)
            data = self.df
            plt.figure(figsize=(8,5))
            plt.scatter(data.zipcode,data.price)
            plt.title("Price Distribution across ZipCodes")
            plt.savefig(plot_path+'/house_price_by_zipcode.png')
            print('Plotting Graph 4')
            plt.pause(5)
        except:
            pass

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Call the training Node after all the Analysis gets completed.

        """
        print('Starting the Model Training Process!')
        print('Model will be trained on the data file present in {} directory'.format(cwd))
        self.next(self.train)

    @step
    def train(self):
        """
        Training a RandomForestRegressor Model and Predicting the House Price based on the inputs given by the user.

        """
        data = self.df
        random.seed(42)
        labels = data['price']
        conv_dates = [1 if values == 2014 else 0 for values in data.date ]
        data['date'] = conv_dates
        train1 = data.drop(['id', 'price'],axis=1)

        from sklearn.model_selection import train_test_split
        x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)

        from sklearn import ensemble
        clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
                  learning_rate = 0.1, loss = 'ls')

        clf.fit(x_train, y_train)
        print('Model Training Complete')
        print('Model Accuracy is: {}'.format(round(clf.score(x_test,y_test),2)))

        bedrooms = self.bedrooms
        bathrooms = self.bathrooms
        sqft_living = self.sqft_living

        if bedrooms <=0:
            bedrooms = 1
            print('Bedrooms cant be less than or equal to 0. Taking default value = 1 Bedroom')
        if bathrooms <=0:
            bathrooms = 1
            print('Bathrooms cant be less than or equal to 0. Taking default value = 1 Bathroom')
        if sqft_living <= 100:
            sqft_living = 100
            print('Sorry! There are no houses having that small living area. Taking default value = 100 Square Ft')

        data_to_predict_on = x_test.loc[x_test['bedrooms'] == bedrooms]
        data_to_predict_on.loc[:,'bathrooms'] = int(bathrooms)
        data_to_predict_on.loc[:,'sqft_living'] = int(sqft_living)

        price = clf.predict(data_to_predict_on)
        self.house_price = round(price[0],2)

        self.next(self.end)

    @step
    def end(self):
        price = self.house_price
        print('Estimated price of the house is ${} '.format(price))

if __name__ == '__main__':
       MetaFlowDemo() 
