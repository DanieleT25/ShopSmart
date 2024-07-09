import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
import os

class DataHandler:
    COLUMNS = ['db_sc', 'id_sc', 'pv', 'data', 'cassa', 'cassiere', 'num_scontrino', 'ora', 'tessera', 't_flag', 'num_riga',
               'r_reparto_cdaplus', 'r_ean', 'r_qta_pezzi', 'r_peso', 'r_importo_lordo', 'r_imponibile', 'r_iva', 'r_sconto',
               'r_sconto_fide', 'r_sconto_rip', 'r_tipo_riga', 'cod_prod', 'descr_prod', 'cat_mer', 'cod_forn', 'descr_forn',
               'liv1', 'descr_liv1', 'liv2', 'descr_liv2', 'liv3', 'descr_liv3', 'liv4', 'descr_liv4', 'liv5', 'descr_liv5',
               'liv6', 'descr_liv6', 'tipologia', 'descr_tipologia', 'cod_rep', 'descr_rep']

    QUARTERS = {
        'Gen-Mar': (1, 3),
        'Apr-Giu': (4, 6),
        'Lug-Set': (7, 9),
        'Ott-Dic': (10, 12)
    }

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        
    def __filter_data(self):
        """Filter the data based on specific conditions"""
        
        # self.df = self.df.dropna(subset=['cod_prod'])
        self.df = self.df.dropna(subset=['liv3'])
        self.df = self.df[self.df['r_tipo_riga'] != 'ANNULLO']
        self.df = self.df[self.df['r_importo_lordo'] != 0]
        self.df = self.df[self.df['cat_mer'] != 'NNNNN']
        self.df = self.df[self.df['cod_prod'] != '1090011']  # shopping bag
    
    def minMax_normalization_1_10(self, df):
        """Normalize the data using Min-Max normalization to the range [1, 10]"""
        
        tmp = df.copy()
        tmp['value'] = 1 + (tmp['value'] - tmp['value'].min()) * 9 / (tmp['value'].max() - tmp['value'].min())
        return tmp

    def split_and_save_data(self, df, path, tSize=0.25):
        """Split the data into training and test sets and save them as CSV files"""

        np.random.seed(42)
        random.seed(42)

        train_data, test_data = train_test_split(df, test_size=tSize)

        # Ensure the path ends with a separator
        if not path.endswith(os.path.sep):
            path += os.path.sep

        # Save in the path
        self.df.to_csv(os.path.join(path, 'data_clean.csv'), index=False)        
        train_data.to_csv(os.path.join(path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(path, 'test_data.csv'), index=False)


    def preprocess_data(self):
        """Preprocess the data"""

        self.df.columns = self.COLUMNS
        
        self.df['data'] = pd.to_datetime(self.df['data'])
        self.df['ora'] = pd.to_datetime(self.df['ora'], format='%H:%M').dt.time
        # self.df['cod_prod'] = self.df['cod_prod'].astype(str)
        # self.df['cod_prod'] = self.df['cod_prod'].apply(lambda x: x.split('.')[0] if '.0' in x else x)

        self.__filter_data()

    def filter_month(self, quarter):
        """Filter the data by the given quarter"""

        if quarter not in self.QUARTERS:
            raise ValueError(f"Quarter '{quarter}' not found. Available quarters: {list(self.QUARTERS.keys())}")

        start_month, end_month = self.QUARTERS[quarter]
        self.df = self.df[(self.df['data'].dt.month >= start_month) & (self.df['data'].dt.month <= end_month)]
        print(f"Dataframe filtered by quarter: {quarter}")

    def remove_outliers(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    def process(self, index, columns, descrArticolo, lamb=False, remuve_outliers=False):
        """Process the data by the given index"""

        if index not in ['tessera', 'id_sc']:
            raise ValueError("index must be 'tessera' or 'id_sc'")
        
        tmp = self.df.groupby([index, columns, descrArticolo]).agg(value=(columns, 'count')).reset_index()

        if lamb is not False:
            tmp = tmp[tmp['value'] <= lamb]
        if remuve_outliers is True:
            tmp = self.remove_outliers(tmp, 'value')
        return tmp

    def get_data(self):
        return self.df
    
    def descriptiveStats(self, df):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df['value'], kde=True)
        plt.title('Purchase Frequency Distribution')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df['value'])
        plt.title('Purchase Frequency Box Plot')

        print("Original Descriptive Statistics:")
        print(df['value'].describe())
    
    def distribSales(self):
       """Plot the distribution of sales by product"""
       
       product_sales = self.df['cod_prod'].value_counts()
       
       plt.figure(figsize=(10, 6))
       product_sales.head(20).plot(kind='bar')
       plt.title('Top 20 Products by Number of Sales')
       plt.xlabel('Category Code')
       plt.ylabel('Number of Sales')
       plt.show()
       
    def freqCustPurch(self):
        """Plot the frequency of customers by number of purchases"""
        
        customer_frequency = self.df['tessera'].value_counts()

        plt.figure(figsize=(10, 6))
        customer_frequency.head(20).plot(kind='bar')
        plt.title('Top 20 Customers by Number of Purchases')
        plt.xlabel('Customer Card ID')
        plt.ylabel('Number of Purchases')
        plt.show()

    def plot_sales(self):
        """Plot weekly sales"""

        self.df['data'] = pd.to_datetime(self.df['data'])
        self.df['day_of_week'] = self.df['data'].dt.dayofweek

        self.df['day_of_week'] = self.df['day_of_week'].apply(lambda x: x + 1)
        weekly_sales = self.df.groupby('day_of_week')['r_importo_lordo'].sum()

        plt.figure(figsize=(10, 6))
        weekly_sales.plot(kind='bar')
        plt.title('Weekly Sales')
        plt.xlabel('Day of the Week')
        plt.ylabel('Gross Sales Amount')
        plt.show()