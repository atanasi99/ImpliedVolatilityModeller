#Author: Atanasije Lalkov
# ---------------------------------------------------------------------
#Description: Creating a portfolio for options connected to MySQL database

import pandas as pd 
import mysql.connector
from mysql.connector.errors import Error
from bs_library import BSE


class MySqlFuncs:

    def __init__(self, usr, pswd, database_name, table_name):
        self.usr = usr
        self.pswd = pswd 
        self.database_name = database_name
        self.table_name = table_name

    def db_connect(self):
        """
        Method for connecting to MySQL
        """
        try:
            # Establish connection to MySQL database
            connect = mysql.connector.connect(
                user = self.usr, 
                password = self.pswd,
                database = self.database_name
                )

            self.connect = connect

            #Create cursor
            self.cursor = connect.cursor()

            # Use db connect to 
            print("Connection to MySQL has been established")

        except Error:
            print(str(Error("Error: Cannot connect to MySQL")))

    def db_close(self):
        # Close connection to database
        self.connect.close()
        print("Connection closed")

    def retrieve_data(self): 
        """
        Method for retrieving data from DB 
        """
        self.cursor.execute("select * from options_table")
        result = self.cursor.fetchall()
        df_queried = pd.DataFrame(result)
        df_queried.columns = ["option_id","option_type","option_ticker","option_price",
        "TTM" ,"strike_price","spot_price","risk_free_rate","delta_value","implied_vol"]

        last_id = df_queried['option_id'].iloc[-1]
        self.options_id = last_id
        print("id = " ,last_id)

        return(df_queried)

    def insert_data(self,df):
        """
        Method for inserting data to DB
        """
        df_len = self.retrieve_data()
        
        option_id = self.generate_option_id()
        obj_BSE = BSE(df.option_type[0],df.spot_price[0],df.TTM[0],df.strike_price[0],df.risk_free_rate[0],df.implied_vol[0])
        delta_value = obj_BSE.delta()
        
        insert_data_string = f"insert into options_table values \
                ({option_id},'{df.option_type[0]}','{df.option_ticker[0]}',{df.option_price[0]},{df.TTM[0]}, \
                {df.strike_price[0]},{df.spot_price[0]},{df.risk_free_rate[0]},{delta_value},{round(df.implied_vol[0],3)});"

        # Execute mysql command 
        self.cursor.execute(insert_data_string)
        # Commit change to database
        self.connect.commit()

    def delete_data(self, options_id):
        delete_string = f"delete from options_table where option_id = {options_id}"
        # Execute mysql command 
        self.cursor.execute(delete_string)
        # Commit change to database
        self.connect.commit()

    def generate_option_id(self):
        self.options_id += 1
        return(self.options_id)

    def update_data(self):
        pass



