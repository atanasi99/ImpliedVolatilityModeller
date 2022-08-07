#Author: Atanasije Lalkov
# ---------------------------------------------------------------------
#Description: Script for retrieving crypto options data from exchange  

# Import required Modules
import asyncio
import json
import websockets
import pandas as pd
from dateutil import parser
from datetime import datetime


class Crypto:
    """
    Class for fetching crypto data from exchange  

    Attributes:
        ticker - ticker of crypto currency 
    Returns:
        - 
    """
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        # Condition check for inputted ticker
        if self.ticker not in ['BTC','ETH']:
            raise ValueError("Invalid error: selected either 'BTC' or 'ETH'")
        self.options_data = None
        self.get_options_data()
        

    @staticmethod
    async def request_API(message):
        """
        Asynchrnous Method for requesting data from API
        Args:
            message: message containing request information
        Return:
            response: response from exchange after request 
        """
        async with websockets.connect('wss://www.deribit.com/ws/api/v2') as ws:
            await ws.send(message)
            while ws.open:
                response = await ws.recv()
                return response

  
    @staticmethod
    def async_message(message):
        return asyncio.get_event_loop().run_until_complete(Crypto.request_API(message))


    def get_options_data(self):
        """
        Method for requesting crypto option chains given specific currency 
        - Cleaning crypto data to be used for imp vol calculations
        """
        msg = \
            {
                "jsonrpc": "2.0",
                "id": 9344,
                "method": "public/get_book_summary_by_currency",
                "params": {
                    "currency": self.ticker,
                    "kind": "option"
                }
            }

        # json response from exchange
        response = self.async_message(json.dumps(msg))

        # Convert json response to data frame
        response = json.loads(response)
        results = response['result']
        df = pd.DataFrame(results)

        # Cleaning dataframe 
        df_cleaned = pd.DataFrame(columns = ['Option_type','Maturity_dates','Strike','Underlying'])

        # Maturity dates, Strike Price and option_type
        for i in df.index:

            instrument_data_i = (df.loc[i,'instrument_name']).split('-')

            # Maturity dates
            maturity = instrument_data_i[-3]
            parsed_maturity = parser.parse(maturity)
            cleaned_maturity = parsed_maturity.strftime('%Y-%m-%d')

            # Strike Price
            cleaned_strike = float(instrument_data_i[-2])
            
            # Option Type
            if instrument_data_i[-1] == 'C':
                cleaned_opt_type = "call"
            if instrument_data_i[-1] == 'P':
                cleaned_opt_type = "put"

            # Underlying price (spot price)
            cleaned_spot = float(df.loc[i,'underlying_price'])

            # Option market data price (using market price)
            cleaned_market_price = float(df.loc[i,'mark_price']) * cleaned_spot

            # Add to newly created dataframe
            df_cleaned = \
            df_cleaned.append(
                            {
                                'Underlying':cleaned_spot,
                                'Option_type':cleaned_opt_type,
                                'Maturity_dates':cleaned_maturity,
                                'Strike':cleaned_strike,
                                'Option_market_price':cleaned_market_price
                            },
                                ignore_index=True
                            )
            # Sort dataframe by maturity dates
            df_cleaned = df_cleaned.sort_values('Maturity_dates')
        self.options_data = df_cleaned.copy(deep=True)

    def get_option_expiries(self):
        """
        Method to get all expiries for given crypto currency 
        Args:
            self
        Return:
            maturities: all maturities for given crypto currency 
        """
        maturities = self.options_data.Maturity_dates.unique()
        return(maturities)

    def get_strike_info(self,maturity_date,type_option):
        """
        Method to get all strikes for given maturity date and option type
        Args:
            maturity_date: option's maturity date
            type_option: call or put
        Return:
            strikes: all strikes for given crypto currency 
        """
        df_strike = self.options_data.copy(deep=True)
        
        df_strike_queried = df_strike.loc[(df_strike['Maturity_dates']==maturity_date) & (df_strike['Option_type']==type_option)]
        df_strike_queried = df_strike_queried.sort_values('Strike')
        strikes = list(df_strike_queried['Strike'])
        return(strikes)

    def spot_price(self,maturity_date,type_option,strike):
        """
        Method to get spot price of crypto
        Args:
            maturity_date: option's maturity date
            type_option: call or put
            strike: strike of specified option
        Return:
            spot: spot price of crypto
        """
        df_spots = self.options_data.copy(deep=True)

        df_spot_queried = \
            df_spots.loc[
                (df_spots['Maturity_dates']==maturity_date) & 
                (df_spots['Option_type']==type_option) &
                ((df_spots['Strike']==strike))
            ]
        
        spot = float(df_spot_queried['Underlying'])
        return(spot)


    def option_market_price(self,maturity_date,type_option,strike):
        df_market = self.options_data.copy(deep=True)

        df_market_queried = \
            df_market.loc[
                (df_market['Maturity_dates']==maturity_date) & 
                (df_market['Option_type']==type_option) &
                ((df_market['Strike']==strike))
            ]
        
        market_price = float(df_market_queried['Option_market_price'])
        return(market_price)



