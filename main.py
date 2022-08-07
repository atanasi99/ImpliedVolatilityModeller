#Author: Atanasije Lalkov
# ---------------------------------------------------------------------
#Description: Newton-Raphson Method to compute implied volatility using market data 

# Import required modules 
import numpy as np 
from datetime import date, datetime
import matplotlib.pyplot as plt
import pandas as pd
# Import local libraries 
import Newton_Raphson_Method
import market_data
import deribit_market_data
import analytics
from analytics import time_to_maturity
from Newton_Raphson_Method import Newton_Raphson
import fitted_vol_analytics
import options_portfolio
import warnings
warnings.filterwarnings("ignore")


def manual_newton_method():
    """
    Newton Raphson method for manual configuration  
    """
    option_type = str(input("Select: call or put: "))
    K = float(input("Enter strike price: "))
    S0 = float(input("Enter spot price: "))
    r = float(input("Enter risk free rate: "))
    market_option_price = float(input("Enter market price of option: "))

    t_question = input("Select 'T' to enter annualised time to maturity or 'D' for maturity date: ")
    if t_question == "T":
        T = float(input("Enter annualised time to maturity: "))
    elif t_question == "D":
        date_T = str(input("Enter maturity date in format YYYY-MM-DD: "))
        T = time_to_maturity(date_T)

    option_ticker = 'manual'

    implied_volatility = Newton_Raphson(option_type,market_option_price,S0,T,K,r)
    
    df = pd.DataFrame({"option_type":option_type, "option_ticker": option_ticker,
                       "option_price":market_option_price , "TTM": T,"strike_price":K, 
                       "spot_price":S0, "risk_free_rate":r, "implied_vol":implied_volatility},index = [0])
    return(df)

def eq_ind_market_newton_method():
        """
        Newton Raphson method for market data values (equity and indices)  
        """
        # Retrieve risk-free rate using risk-free class from market data class
        r = market_data.risk_free_rates().SONIA_rfr()

        print("Equity ticker examples: AAPL, AMZN, TSLA, GS, etc.")
        print("Index ticker examples: ^SPX\n")

        Ticker = str(input("Enter Security ticker: "))
        # Create object of ImpVol class
        model = market_data.equity_indices(Ticker)
        # Output the maturities of selected security
        print(model.get_option_expiries())
        
        # Select maturity from available maturity dates
        maturity_date = str(input("Enter maturity date of option: "))
        T = time_to_maturity(maturity_date)

        option_type = str(input("Enter type of European option either 'call' or 'put': "))
        
        # Observe strikes and strike interval
        strike_info = model.get_strike_info(maturity_date,option_type)
        print(strike_info,"\n")
        K = float(input("Enter strike: "))

        S0 = model.spot_price()
        market_option_price = model.option_market_price(maturity_date,option_type,K)

        print(f"\n\nSpot price: {S0} \nMarket price: {market_option_price} \nTTM: {T} \nK: {K}")

        implied_volatility = Newton_Raphson(option_type,market_option_price,S0,T,K,r)

        df = pd.DataFrame({"option_type":option_type, "option_ticker": Ticker,
                       "option_price":market_option_price , "TTM": T,"strike_price":K, 
                       "spot_price":S0, "risk_free_rate":r, "implied_vol":implied_volatility},index = [0])

        return(df)


def crypto_market_newton_method():
        """
        Newton Raphson method for market data values (crypto)  
        """
        # Retrieve risk-free rate using risk-free class from market data class
        r = market_data.risk_free_rates().SONIA_rfr()

        print("Crypto tickers available: 'BTC' or 'ETH\n")

        Ticker = str(input("Enter Security ticker: "))
        # Create object of ImpVol class
        model = deribit_market_data.Crypto(Ticker)
        # Output the maturities of selected security
        print(model.get_option_expiries())
        
        # Select maturity from available maturity dates
        maturity_date = str(input("Enter maturity date of option: "))
        T = time_to_maturity(maturity_date,"Crypto")

        option_type = str(input("Enter type of European option either 'call' or 'put': "))
        
        # Observe strikes and strike interval
        strike_info = model.get_strike_info(maturity_date,option_type)
        print(strike_info,"\n")
        K = float(input("Enter strike: "))

        S0 = model.spot_price(maturity_date,option_type,K)
        market_option_price = model.option_market_price(maturity_date,option_type,K)

        print(f"\n\nSpot price: {S0} \nMarket price: {market_option_price} \nTTM: {T} \nK: {K}")

        implied_volatility = Newton_Raphson(option_type,market_option_price,S0,T,K,r)

        df = pd.DataFrame({"option_type":option_type, "option_ticker": Ticker,
                       "option_price":market_option_price , "TTM": T,"strike_price":K, 
                       "spot_price":S0, "risk_free_rate":r, "implied_vol":implied_volatility},index = [0])

        return(df)


def eq_ind_volatility_smiles():
    """
    volatility smile method for computing smiles for equity/indices 
    """

    print("Equity ticker examples: AAPL, AMZN, TSLA, GS, etc.")
    print("Index ticker examples: ^SPX\n")


    Ticker = str(input("Enter Security ticker: "))
    # Create object of ImpVol class
    obj_market_data = market_data.equity_indices(Ticker)
    obj_smile = analytics.VolatilitySmile(Ticker)
    S0 = obj_market_data.spot_price()

    flag_exit = True
    maturities = list()
    option_type = str(input("Enter type of European option either 'call' or 'put': "))
    # Output the maturities of selected security
    print(obj_market_data.get_option_expiries())
    while flag_exit == True:
        condition = str(input("Enter maturity date or type 'exit' to finish\nEnter: "))
        if condition == "exit":
            break
        elif condition != "exit":
            maturities.append(condition)

    option_dfs = list()

    for maturity in maturities:
        # market price is set to midPrice currently
        df = obj_smile.compute_smile(maturity,option_type)
        # Apply SVI paramterization
        df = df.dropna(axis= 0, how='any')
        # Remove options that have a market price of 0
        df.drop(df[df['Market price'] == 0.0].index)

        # Optimising using SVI
        obj_SVI = fitted_vol_analytics.SVIModel(df,S0)
        df_updated = obj_SVI.SVI_fit()

        option_dfs.append(df_updated)

    fig, ax = plt.subplots()
    
    for i in range(len(option_dfs)):
        df_current = option_dfs[i]
        df_current.plot(ax=ax,x='Strike',y='fitted_imp_vols',label = maturities[i])

    # Plot ATM price
    plt.axvline(x=S0,color = 'r',ls='--',label = "ATM")

    plt.legend()
    plt.title(f"Volatility smile for {Ticker} {option_type}")
    plt.ylabel('Implied Volatility')
    plt.xlabel('Strike price')
    plt.grid()
    plt.show()

    
def crypto_volatility_smiles():
    """
    volatility smile method for computing smiles for cryptos
    """
    print("Crypto tickers available: 'BTC' or 'ETH\n")

    Ticker = str(input("Enter Security ticker: "))
    # Create object of ImpVol class
    obj_market_data = deribit_market_data.Crypto(Ticker)
    obj_smile = analytics.VolatilitySmile(Ticker)
    
    flag_exit = True
    maturities = list()
    option_type = str(input("Enter type of European option either 'call' or 'put': "))
    print(obj_market_data.get_option_expiries())
    while flag_exit == True:
        condition = str(input("Enter maturity date or type 'exit' to finish\nEnter: "))
        if condition == "exit":
            break
        elif condition != "exit":
            maturities.append(condition)
    
    option_dfs = list()
    for maturity in maturities:
        df_maturity = obj_smile.compute_smile_crypto(maturity,option_type)
        df_maturity = df_maturity.dropna(axis= 0, how='any')

        # Optimising using kernel 
        obj_kernel = fitted_vol_analytics.KernelRegModel(df_maturity)
        df_updated = obj_kernel.fit_gauss_kernel_reg()

        option_dfs.append(df_updated)
    
    fig, ax = plt.subplots()
    for i in range(len(option_dfs)):
        df = option_dfs[i]
        df.plot(ax=ax,x='Strike',y='fitted_imp_vols',label = maturities[i])

    plt.legend()
    plt.title(f"Volatility smile for {Ticker} {option_type}")
    plt.ylabel('Implied Volatility')
    plt.xlabel('Strike price')
    plt.grid()
    plt.show()


def crypto_surface_plot(action_type):
    """
    Method for computing crypto implied vol surface
    """
    print("Crypto tickers available: 'BTC' or 'ETH\n")
    Ticker = str(input("Enter Security ticker: "))
    # Create object of ImpVol class
    obj_market_data = deribit_market_data.Crypto(Ticker)
    obj_surface = analytics.VolatilitySurface(Ticker)
    
    flag_exit = True
    maturities = list()
    option_type = str(input("Enter type of European option either 'call' or 'put': "))
    expirations = obj_market_data.get_option_expiries()
    print(expirations)
    while flag_exit == True:
        condition = str(input("Enter maturity date or type 'exit' to finish\nEnter: "))
        if condition == "exit":
            break
        elif condition != "exit":
            maturities.append(condition)

    result_df = obj_surface.compute_surface(maturities,option_type,action_type)
    obj_surface.plot_surface(result_df)

4
def eq_ind_surface_plot(action_type):
    """
    Method for computing equity implied vol surface
    """
    print("Equity ticker examples: AAPL, AMZN, TSLA, GS, etc.")
    print("Index ticker examples: ^SPX\n")
    Ticker = str(input("Enter Security ticker: "))
    # Create object of ImpVol class
    obj_market_data = market_data.equity_indices(Ticker)
    obj_surface = analytics.VolatilitySurface(Ticker)

    flag_exit = True
    maturities = list()
    option_type = str(input("Enter type of European option either 'call' or 'put': "))
    # Output the maturities of selected security
    print(obj_market_data.get_option_expiries())
    while flag_exit == True:
        condition = str(input("Enter maturity date or type 'exit' to finish\nEnter: "))
        if condition == "exit":
            break
        elif condition != "exit":
            maturities.append(condition)

    result_df = obj_surface.compute_surface(maturities,option_type,action_type)
    obj_surface.plot_surface(result_df)


def fitted_vol_analysis():
    """
    Method for computing fitted volatility analytics
    """
    print("Only equity or index tickers are available")
    print("Equity ticker examples: AAPL, AMZN, TSLA, GS, etc.")
    print("Index ticker examples: ^SPX\n")
    Ticker = str(input("Enter Security ticker: "))
    # Create object of ImpVol class
    obj_market_data = market_data.equity_indices(Ticker)

    option_type = str(input("Enter type of European option either 'call' or 'put': "))
    # Output the maturities of selected security
    print(obj_market_data.get_option_expiries())
    maturity_date = str(input("Enter maturity date: "))
    obj_fitted = fitted_vol_analytics.fittedVolAnalysis(Ticker, maturity_date,option_type)
    # Computed fitted implied volatilities for all methods
    obj_fitted.compute_fitted_methods()
    obj_fitted.fitted_methods_plot()
    obj_fitted.compute_fitted_metrics()

def option_portfolio_page(action):
    """
    Method for interacting with MySQL db
    """
    usr = 'root'
    pswd = 'Implied1#'
    db_name = 'option_portfolio'
    tb_name = 'options_table'
    obj_mysql = options_portfolio.MySqlFuncs(usr, pswd, db_name, tb_name)
    
    if action == "initialise":
        # connect to mysql database
        obj_mysql.db_connect()
        # Retrieve data
        df = obj_mysql.retrieve_data()
        print(df)

    if action == "insert_market":
        obj_mysql.db_connect()

        asset_type = (str(input("Select type of asset \n- 'Equity' \n- 'Index' \n- 'Crypto'\nEnter Choice: "))).title()

        if asset_type.title() in ["Equity","Index"]:
            df = eq_ind_market_newton_method()

        if asset_type == "Crypto":
            df = crypto_market_newton_method()

        obj_mysql.insert_data(df)
        print("data has been inserted")

        df_queried = obj_mysql.retrieve_data()
        print(df_queried)

    if action == "insert_manual":
        obj_mysql.db_connect()

        df = manual_newton_method()

        obj_mysql.insert_data(df)
        print("data has been inserted")

        df_queried = obj_mysql.retrieve_data()
        print(df_queried)

    if action == "delete": 
        obj_mysql.db_connect()
        
        delete_id = int(input("Enter option_id of option to be deleted: "))
        obj_mysql.delete_data(delete_id)
        print("Option data deleted")
        df = obj_mysql.retrieve_data()
        print(df)

    if action == 'close':
        obj_mysql.db_close()



# Main execution
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    exit_flag = False
    print("Welcome to Implied Volatility Calculator\nat anytime press Ctrl-C to exit")

    while exit_flag != True:
        try:
            action_type = int(input("\nMenu: \
            \n\nSelect '1' for implied volatility calculator \
            \nSelect '2' for implied volatility smile  \
            \nSelect '3' for implied volatility surface \
            \nSelect '4' for fitted volatility analysis \
            \nSelect '5' for options portfolio \
            \nSelect '6' to exit program \
            \nEnter: "))

            print("\n")
            
            if action_type == 1:
                #Type of configuration 
                Configuration = str(input("Enter 'manual' or 'market' configuration: "))

                if Configuration == "manual":
                    df_imp_vol = manual_newton_method()
                    print(f"Implied volatility is: {df_imp_vol.implied_vol[0]}")

                if Configuration == "market":
                    asset_type = (str(input("Select type of asset \n- 'Equity' \n- 'Index' \n- 'Crypto'\nEnter Choice: "))).title()
                    
                    if asset_type.title() in ["Equity","Index"]:
                        df_market_eq = eq_ind_market_newton_method()
                        print(f"Implied volatility is: {df_market_eq.implied_vol[0]}")

                    if asset_type == "Crypto":
                        df_market_crypto = crypto_market_newton_method()
                        print(f"Implied volatility is: {df_market_crypto.implied_vol[0]}")

            if action_type == 2:
                asset_type = (str(input("Select type of asset \n- 'Equity' \n- 'Index' \n- 'Crypto'\nEnter Choice: "))).title()
                
                if asset_type.title() in ["Equity","Index"]:
                    eq_ind_volatility_smiles()
                    
                if asset_type.title() == "Crypto":
                    crypto_volatility_smiles()


            if action_type == 3:
                asset_type = (str(input("Select type of asset \n- 'Equity' \n- 'Index' \n- 'Crypto'\nEnter Choice: "))).title()

                if asset_type.title() in ["Equity","Index"]:
                    eq_ind_surface_plot(asset_type)
                    
                if asset_type.title() == "Crypto":
                    crypto_surface_plot(asset_type)

            if action_type == 4:
                # for now just equity
                fitted_vol_analysis()

            if action_type == 5:
                # retrieve options data
                option_portfolio_page('initialise')

                portfolio_action = int(input("Select action \n1. Insert market option '1' \n2. Insert option manually '2' \n3. Delete option from portfolio '3' \n4. Exit to home '4'\nEnter Choice: "))

                if portfolio_action == 1:
                    option_portfolio_page('insert_market')

                if portfolio_action == 2:
                    option_portfolio_page('insert_manual')

                if portfolio_action == 3:
                    option_portfolio_page('delete')

                if portfolio_action == 4:
                    continue

            if action_type == 6:
                print("\nProgram exited")
                # end session with mysql
                #option_portfolio_page('close')
                break
                    
        except KeyboardInterrupt:
            print("\nProgram exited")
              



