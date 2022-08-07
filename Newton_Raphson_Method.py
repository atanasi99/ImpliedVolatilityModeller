#Author: Atanasije Lalkov
# ---------------------------------------------------------------------
#Description: Newton-Raphson Method to compute implied volatility 

# Import required modules
import numpy as np

# Import local modules from other scripts 
from bs_library import BSE

def Newton_Raphson(option_type,market_option_price,S0,T,K,r,D=0,sigma_old=1,tol=0.001):
        """
        Newton Raphson method for computing implied volatility
        Args:
            opt_type - type of european option (call/put)
            market_option_price - market price of the option
            S0 - Spot price (underlying)
            T - Time to maturity
            K - Strike Price
            r - risk free rate 
            sigma - volatility
            D - Dividends 
            tol - tolerance
            sigma_old - initial sigma
        Return:
            implied_vol: implied volatility
        """
        # Maximum iterations
        max_iter = 10000
        # Initialise counter 
        i = 1
        # Iteration to determine imp vol 
        while i < max_iter:
            # Create object used by methods in BS library
            Bs = BSE(option_type,S0,T,K,r,sigma_old)
            # Compute option price
            bs_price = Bs.option_price()
            
            # Compute vega
            fprime = Bs.vega()

            # Check for vega = 0 
            if fprime == 0:
                # Implied volatility is 0
                implied_vol = np.NaN
                return(implied_vol)

            # Compute f(x) which is different in prices
            fx =  bs_price - market_option_price
            
            # Use newton Raphson equation
            sigma_new = sigma_old - fx/fprime
            
            # Conditional statement if abs difference between vol is less than tolerance
            if np.abs(sigma_old - sigma_new) < tol :
                # If condition true break
                break
            # If condition is not met continue with iteration
            i = i + 1
            sigma_old = sigma_new

        # Imp Vol is updated volatility 
        implied_vol = sigma_new
        # Return Implied Volatility
        return(implied_vol)


        

