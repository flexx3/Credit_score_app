import pandas as pd
import sqlite3

class load_data:
    def __init__(self, connection):
        self.connection= connection
    #Extract and transform data from csv file
    def _load_csv(self, filepath):
        data= pd.read_csv(filepath, index_col=0, na_values='')
        #convert object column dtypes to category
        object_columns= data.select_dtypes(include='object').columns
        data[object_columns]= data[object_columns].astype('category')
        self.data= data
    #load data into database
    def _insert_into_db(self, table_name, if_exists='replace'):
        inserted_data= self.data.to_sql(name=table_name, con=self.connection, if_exists=if_exists)
        return{'Transaction successful ': True,
              'no of records ': inserted_data}
    #load the data from database
    def read_data_from_db(self, table_name, index_col=0):
        query= F"""SELECT * FROM  {table_name} """
        df= pd.read_sql(query, con=self.connection)
        return df
    
    
        
        
        

