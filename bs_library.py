#Author: Atanasije Lalkov
# ---------------------------------------------------------------------
#Description: BLack-Scholes library used for Newton-Raphson method for computing implied volatility

# Import required Modules
import numpy as np 
from scipy import stats


class BSE():
    """
    Class for computing price of European call and put options using BSE

    Attributes:
        opt_type - type of European option (call/put)
        S0 - Spot price (underlying)
        T - Time to maturity
        K - Strike Price
        r - risk free rate 
        sigma - volatility
        D - Dividends 
    Returns:
        - 
    """
    def __init__(self,opt_type,S0,T,K,r,sigma,D=0):
        self.opt_type = opt_type
        self.S0 = float(S0)
        self.T = T
        self.K = K
        self.r = r
        self.D = D
        self.sigma = sigma

    def option_price(self):
        """
        Method for computing Black Scholes European call/put value
        Args:
            self
        Return:
            price: price of the option
        """

        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - (self.sigma) * np.sqrt(self.T)

        # If option type is call
        if self.opt_type == "call":
            # Computes option value
            price = (self.S0 * np.exp(-self.D * self.T) * stats.norm.cdf(d1)) - (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2))
        # If option type is put
        elif self.opt_type == "put":
            # Computes option value
            price = (-self.S0 * np.exp(-self.D * self.T) * stats.norm.cdf(-d1)) + (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2))

        # Return option price 
        return(price)

    def vega(self):
        """
        Method for computing vega
        Args:
            self
        Return:
            vega_value: value of vega for specified option
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))

        # Computes vega value
        vega_value = self.S0 * np.sqrt(self.T) * stats.norm.pdf(d1)
        # Returns vega
        return(vega_value)

    def delta(self):
        """
        Method for computing the greek delta
        Args:
            self
        return:
            delta_value: value of the greek delta
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta_value = np.exp(-self.D * self.T) * stats.norm.cdf(d1)
        return(delta_value)



