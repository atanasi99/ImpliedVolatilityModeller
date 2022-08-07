#Project Code
#Author: Atanasije Lalkov
# ---------------------------------------------------------------------
#Description: Script for retrieving market data required for Implied Volatility 

# Import required modules 
from operator import index
import numpy as np 
import pandas as pd
import yfinance as yf 
import requests     
import io
from datetime import datetime, timedelta


class equity_indices():
    """
    Class for retrieving equity and indices market data from Yahoo Finance API 

    Attributes:
            ticker: the tick code of the security 
    """
    def __init__(self,ticker):
        # Ticker of security
        self.ticker = ticker 
        # Object for data of security 
        self.asset = yf.Ticker(self.ticker)

    def get_option_expiries(self):
        """
        Method for retrieving all maturity dates of listed options for specified security 
        Args:
            self
        Return:
            asset_maturities: list of maturity dates for specified securities 
        """
        # From asset data retrieve option maturity dates
        asset_maturities = self.asset.options
        # Output the maturity dates of the asset options
        return(asset_maturities)

    def get_strike_info(self,maturity_date,type_option):
        """
        Method for retrieving all strike prices for specified maturity
        Args:
            maturity_date: is expiry date of option 
            option_type: type of European option (call/put)
        Return:
            strike_df: dataframe of all strike prices
        """
        # Select options for selected maturity
        opts = self.asset.option_chain(maturity_date)
        # Select whether option is call or put
        if (type_option == 'call'):
            opt = opts.calls
        elif (type_option == 'put'):
            opt = opts.puts
        
        strike_min = opt['strike'].min()
        strike_max = opt['strike'].max()
        interval = opt.loc[2]['strike'] - opt.loc[1]['strike']

        strike_df = pd.DataFrame({"Minimum Strike":strike_min,"Maximum Strike":strike_max,"Strike Interval":interval},index=[maturity_date])
        return(strike_df)

    def option_market_price(self,maturity_date,option_type,K,price_type='ask'):
        """
        Method for retrieving market price of option
        Args:
            maturity_date: is expiry date of option 
            option_type: type of European option (call/put)
            K: strike price of option
            price_type: type of market price -> lastPrice, ask, bid or mid-price 
        Return:
            market_price: market price of specified option
        """
        # Select options for selected maturity
        opts = self.asset.option_chain(maturity_date)
        # Select whether option is call or put
        if (option_type == 'call'):
            opt = opts.calls
        elif (option_type == 'put'):
            opt = opts.puts
        #Retrieves option price
        market_price = float(opt.loc[opt['strike'] == K,price_type])
        # Redundancy check as ask can be 0 from API so use lastPrice instead 
        if market_price == 0:
            market_price = float(opt.loc[opt['strike'] == K,'lastPrice'])
        # Returns market price of option
        return(market_price)

    def spot_price(self):
        """
        Method for retrieving spot price of underlying
        Args:
            self
        Return:
            S0: spot price of underlying
        """
        # Retrieve security prices
        security_prices = self.asset.history()
        # Sort dataframe in date descending order 
        security_prices_sorted = security_prices.sort_values('Date',ascending=False)
        # Select current/recent closing price
        security_price = security_prices_sorted[:1]
        # get final closing price of option
        S0 = float(security_price['Close'])
        S0 = np.round(S0,2)
        return(S0)



class risk_free_rates():
    """
    Class for retrieving risk-free rates  
    """
    def __init__(self):
        # Use todays date as attribute for computing any rfr (risk-free rate)
        now = (datetime.today()).date()
        week_ago = now - timedelta(days = 7)

        self.todays_date = now.strftime("%d/%b/%Y")
        self.week_ago = week_ago.strftime("%d/%b/%Y")

    def SONIA_rfr(self):
        """
        Method for retrieving recent SONIA rates for risk-free rate 
        args:
            self
        returns:
            rfr_rate/100: current risk-free rate
        """
        # Define URL endpoint which is IADB
        url_end = "http://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes"

        # Define parameters for our request in payload - use 'IUDSOIA'
        payload = {
            'Datefrom'   : self.week_ago,
            'Dateto'     : self.todays_date,
            'SeriesCodes': 'IUDSOIA',
            'CSVF'       : 'TN',
            'UsingCodes' : 'Y',
            'VPD'        : 'Y',
            'VFD'        : 'N'
        }

        # Use request package to obtain response from IADB
        response = requests.get(url_end, params=payload)
        
        # In response use url to send request to download data
        response_url = response.url

        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/54.0.2840.90 '
                  'Safari/537.36'
        }

        # Use response url to send request for download data
        response_SONIA = requests.get(response_url,headers=headers)

        # The response from the database contains the data from the content 
        # Use io to write and then pandas to read from file
        df_SONIA = pd.read_csv(io.BytesIO(response_SONIA.content))

        # Convert index of dataframe to datetime format 
        df_SONIA['DATE'] = pd.to_datetime(df_SONIA['DATE'])
        df_SONIA = df_SONIA.set_index('DATE')

        # Sort dataframe by date so that latest result is first
        df_sorted = df_SONIA.sort_index(ascending=False)

        rfr_rate = ((df_sorted.iloc[0]).values)[0]

        # Return risk free rate
        return(rfr_rate/100)

