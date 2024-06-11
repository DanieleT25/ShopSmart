import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

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
        
        self.df = self.df.dropna(subset=['cod_prod'])
        self.df = self.df[self.df['r_tipo_riga'] != 'ANNULLO']
        self.df = self.df[self.df['r_importo_lordo'] != 0]
        self.df = self.df[self.df['cat_mer'] != 'NNNNN']
        self.df = self.df[self.df['cod_prod'] != '1090011']  # shopping bag

    def __split_and_save_data(self):
        """Split the data into training and test sets and save them as CSV files"""

        np.random.seed(42)
        random.seed(42)

        train_data, test_data = train_test_split(self.df, test_size=0.25)

        train_data.to_csv('../Data/train_data.csv', index=False)
        test_data.to_csv('../Data/test_data.csv', index=False)

    def preprocess_data(self):
        """Preprocess the data"""

        self.df.columns = self.COLUMNS
        
        self.df['data'] = pd.to_datetime(self.df['data'])
        self.df['ora'] = pd.to_datetime(self.df['ora'], format='%H:%M').dt.time
        self.df['cod_prod'] = self.df['cod_prod'].astype(str)
        self.df['cod_prod'] = self.df['cod_prod'].apply(lambda x: x.split('.')[0] if '.0' in x else x)

        self.__filter_data()

    def filter_month(self, quarter):
        """Filter the data by the given quarter"""

        if quarter not in self.QUARTERS:
            raise ValueError(f"Quarter '{quarter}' not found. Available quarters: {list(self.QUARTERS.keys())}")

        start_month, end_month = self.QUARTERS[quarter]
        self.df = self.df[(self.df['data'].dt.month >= start_month) & (self.df['data'].dt.month <= end_month)]
        print(f"Dataframe filtered by quarter: {quarter}")

    def process(self, index):
        """Process the data by the given index"""

        if index not in ['tessera', 'id_sc']:
            raise ValueError("index must be 'tessera' or 'id_sc'")
        
        self.df = self.df.groupby([index, 'cod_prod', 'descr_prod']).agg(value=('cod_prod', 'count')).reset_index()
        self.__split_and_save_data()

    def get_data(self):
        return self.df
    
    def distribSales(self):
       """Plot the distribution of sales by product"""
       
       product_sales = self.df['cod_prod'].value_counts()
       
       plt.figure(figsize=(10, 6))
       product_sales.head(20).plot(kind='bar')
       plt.title('Top 20 Prodotti per Numero di Vendite')
       plt.xlabel('EAN del Prodotto')
       plt.ylabel('Numero di Vendite')
       plt.show()
       
    def freqCustPurch(self):
        """Plot the frequency of customers by number of purchases"""
        
        customer_frequency = self.df['tessera'].value_counts()

        plt.figure(figsize=(10, 6))
        customer_frequency.head(20).plot(kind='bar')
        plt.title('Top 20 Clienti per Numero di Acquisti')
        plt.xlabel('ID Tessera Cliente')
        plt.ylabel('Numero di Acquisti')
        plt.show()

    def plot_sales(self):
        """Plot monthly and weekly sales"""

        self.df['month'] = self.df['data'].dt.month
        self.df['day_of_week'] = self.df['data'].dt.dayofweek

        monthly_sales = self.df.groupby('month')['r_importo_lordo'].sum()
        plt.figure(figsize=(10, 6))
        monthly_sales.plot(kind='bar')
        plt.title('Vendite Mensili')
        plt.xlabel('Mese')
        plt.ylabel('Importo Lordo delle Vendite')
        plt.show()

        self.df['day_of_week'] = self.df['day_of_week'].apply(lambda x: x + 1)
        weekly_sales = self.df.groupby('day_of_week')['r_importo_lordo'].sum()

        plt.figure(figsize=(10, 6))
        weekly_sales.plot(kind='bar')
        plt.title('Vendite Settimanali')
        plt.xlabel('Giorno della Settimana')
        plt.ylabel('Importo Lordo delle Vendite')
        plt.show()