#Project Code
#Author: Atanasije Lalkov
# ---------------------------------------------------------------------
#Description: Analytics for implied volatility which includes
# 1. Volatility Smiles
# 2. Volatility surfaces 

# Import required libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import cm
import yfinance as yf
from datetime import datetime

# Import local libraries 
import deribit_market_data
import market_data
import Newton_Raphson_Method
from Newton_Raphson_Method import Newton_Raphson


def time_to_maturity(maturity_date,asset_type="Other"):
    """
    Method for computing time to maturity given maturity date  
    Args:
        maturity date - is maturity date of option
    Return:
        T: Time to maturity (Annualised)
    """
    # Retrieve today's date
    current_date  = (datetime.today()).date()
    # Convert expiry date to time format
    expiry_date = (datetime.strptime(maturity_date,"%Y-%m-%d")).date()
    # Compute time to maturity
    T_days = (expiry_date - current_date).days

    # Compute annualised time to maturity 
    if asset_type == "Crypto":
        Trading_days = 365
    else:
        Trading_days = 252

    T = np.round((T_days/Trading_days),decimals = 4)
    return(T)


# Class for computing volatility smiles
class VolatilitySmile():
    # Global variable for ticker
    tick = None

    def __init__(self,ticker):
        self.ticker = ticker
        VolatilitySmile.tick = ticker
        # fetch risk-free rate from BoE
        self.r = market_data.risk_free_rates().SONIA_rfr()
        
    def compute_smile(self,maturity_date,option_type,price_type="ask"):
        """
        Method for computing volatility smile for equity/indices

        Args:
            maturity_date: maturity date for options with different strikes
            
        Returns:
            smile_data_df: dataframe of vol smile data including imp vol 
        """
        self.maturity_date = maturity_date
        self.option_type = option_type
        # Retrieve data for security
        options = yf.Ticker(self.ticker)
        # Retrieve options of expiry date
        options_chain = options.option_chain(maturity_date)
        # Select options regarding the type of option
        if option_type == "call":
            option_CP = options_chain.calls
        if option_type == "put":
            option_CP = options_chain.puts
        
        # Compute number of data points containing different strikes
        num_strikes = len(option_CP.index)
        # Create empty dataframe for smile values
        smile_data_df = pd.DataFrame()

        # Iterate over number of strikes
        for i in range(num_strikes):
            print ("\033[A                             \033[A")
            print(f"Implied Vol being computed: {np.round(((i/num_strikes)*100))}% ")
            # dict to store current option data
            current_option = {'Date':(datetime.today()).date(),
                        'Maturity':maturity_date,'TTM':time_to_maturity(maturity_date),
                        'Strike':None,'Market price':None,'Implied volatility':None}

            # get current option
            opt = option_CP.loc[i]
            # get strike of current option
            K = float(opt['strike'])
            current_option['Strike'] = K

            # get market price
            # ------------------------------------------------
            # Select either ask, bid, lastPrice or midPrice
            if price_type == "midPrice":
                ask = float(opt['ask'])
                bid = float(opt['bid'])
                market_price = (bid+ask)/2
                
            elif price_type in ['ask','bid','lastPrice']:
                market_price = float(opt[price_type])

            current_option['Market price'] = market_price

            # fetch underlying 
            obj = market_data.equity_indices(self.ticker)
            S0 = obj.spot_price()

            # compute implied volatility using imported Newton Algorithm
            imp_vol = Newton_Raphson(option_type,market_price,S0,time_to_maturity(maturity_date),K,self.r)
            current_option['Implied volatility'] = imp_vol

            # Convert dict to pd.dataFrame
            df = pd.DataFrame(current_option,index = [i])
            # Concatenate to options dataframe
            smile_data_df = pd.concat([smile_data_df,df])

        # Return smile df
        return(smile_data_df)



    def compute_smile_crypto(self,maturity_date,option_type):
        """
        Method for computing volatility smile for crypto
        Args:
            maturity_date: maturity date for options with different strikes
            
        Returns:
            smile_data_df: dataframe of vol smile data including imp vol 
        """
        # Retrieve data for crypto currency
        options = deribit_market_data.Crypto(self.ticker)
        option_df = options.options_data.copy(deep=True)

        # Query dataframe for input maturity and option type
        option_df_queried = \
            option_df.loc[
                    ( option_df['Maturity_dates']==maturity_date) & 
                    ( option_df['Option_type']==option_type)
            ]
        #Sort df by strike price
        option_df_queried = option_df_queried.sort_values('Strike')
        # Compute number of data points containing different strikes
        num_strikes = len(option_df_queried.index)
        option_df_queried.index = np.arange(num_strikes)
        
        # Create empty dataframe for smile values
        smile_data_df = pd.DataFrame()

        # Iterate over number of strikes
        for i in range(num_strikes):
            print ("\033[A                             \033[A")
            print(f"Implied Vol being computed: {np.round(((i/num_strikes)*100))}% ")
            # dict to store current option data
            current_option = {'Date':(datetime.today()).date(),
                        'Maturity':maturity_date,'TTM':time_to_maturity(maturity_date,"Crypto"),
                        'Strike':None,'Market price':None,'Implied volatility':None}
            
            # get current option
            opt = option_df_queried.loc[i]
        
            # get strike of current option
            K = float(opt['Strike'])
            current_option['Strike'] = K

            # get market price
            market_price = float(opt['Option_market_price'])
            current_option['Market price'] = market_price

            # fetch underlying 
            S0 = float(opt['Underlying'])

            # compute implied volatility 
            imp_vol = Newton_Raphson(option_type,market_price,S0,time_to_maturity(maturity_date,"Crypto"),K,self.r)
            current_option['Implied volatility'] = imp_vol

            # Convert dict to pd.dataFrame
            df = pd.DataFrame(current_option,index = [i])
            # Concatenate to options dataframe
            smile_data_df = pd.concat([smile_data_df,df])

        return(smile_data_df)


class VolatilitySurface(VolatilitySmile):
    def __init__(self,ticker):
        self.ticker = ticker
        super().__init__(ticker)
        

    def compute_surface(self,maturities,option_type,asset):
        """
        Method for computing volatility surface 
        Args:
            maturities: maturity date for options with different strikes
            option_type: either call or put
            asset: type of asset either equity/index or crypto
        Returns:
            options_dfs: dataframe of vol surface 
        """
        self.option_type = option_type
        # Create dataframe to store surface data
        option_dfs = pd.DataFrame()
        # For each given maturity compute implied volatility for each strike using smile method
        for maturity in maturities:
            if asset.title() == "Crypto":
                df = self.compute_smile_crypto(maturity,option_type)
                option_dfs = pd.concat([option_dfs,df],axis= 0)

            elif asset.title() in ["Equity","Indices"]:
                # market price is set to midPrice currently
                df = self.compute_smile(maturity,option_type)
                df.drop(df[df['Market price'] == 0.0].index)
                option_dfs = pd.concat([option_dfs,df],axis= 0)
        
        return(option_dfs)


    def plot_surface(self, option_dfs):
        """
        Method for computing volatility surface 
        Args:
            options_df: volatility surface df 
        Returns:
            plot surface 
        """
        # pivot the dataframe
        options_pivot = option_dfs[['TTM', 'Strike', 'Implied volatility']].pivot(index='Strike', columns='TTM', values='Implied volatility').dropna()

        # create the figure object
        fig = plt.figure()

        # add the subplot with projection argument
        ax = fig.add_subplot(111, projection='3d')

        # get the 1d values from the pivoted dataframe
        x, y, z = options_pivot.columns.values, options_pivot.index.values, options_pivot.values

        # return coordinate matrices from coordinate vectors
        X, Y = np.meshgrid(x, y)

        # set labels
        ax.set_xlabel('Time to maturity')
        ax.set_ylabel('Strike price')
        ax.set_zlabel('Implied volatility')
        ax.set_title(f'Implied volatility surface of {self.option_type} for {self.ticker.upper()}')

        # plot
        surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0.1, antialiased=False, alpha = 0.8)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


    
