#Author: Atanasije Lalkov
# ---------------------------------------------------------------------
#Analytics of different methods to fit volatility smiles
# 1. DVF
# 2. Kernel Regression 
# 3. SVI 
# 4. Polynomial Regression 

# Import required libraries
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from datetime import datetime

from statsmodels.nonparametric.kernel_regression import KernelReg 
from scipy.interpolate import interp1d
from scipy.optimize import Bounds, minimize

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import auc
from sklearn.metrics import mean_absolute_error

# Import local libraries
import analytics 
import market_data
from bs_library import BSE


# Fitted volatility analytics 
class fittedVolAnalysis:
    
    def __init__(self, ticker, maturity_date,option_type,price_type="midPrice"):
        self.ticker = ticker
        self.maturity_date = maturity_date
        self.option_type = option_type
        self.price_type = price_type
        self.S0 = market_data.equity_indices(ticker).spot_price()
        self.r = market_data.risk_free_rates().SONIA_rfr()
        self.T = analytics.time_to_maturity(maturity_date)

        self.df = None

    def compute_fitted_market_price(self,df):
        """
        Method for computing option price using BSE
        Args:
            df: dataframe with fitted volatility data
        Returns:
            df: dataframe with additional option prices
        """
        df['fittedMarketPrice'] = \
            df.apply(
                lambda iv: 
                BSE(self.option_type,self.S0,self.T,iv.Strike,self.r,iv.fitted_imp_vols).option_price(), 
                axis = 1)
        return(df)
        
    def compute_smile_df(self):
        """
        Method for computing smile of fitted volatility 
        """
        model_smile = analytics.VolatilitySmile(self.ticker)
        df = model_smile.compute_smile(self.maturity_date,self.option_type,self.price_type)
        df = df.dropna(axis= 0, how='any')
        df = df.set_index([np.arange(0,len(df))])
        self.df = df 
       
    def compute_fitted_methods(self):
        """
        Method for computing fitted volatility using different curve-fitting techniques
        """
        self.compute_smile_df()

        # Create instances for fitting models
        DVF_model = DVFModel(self.df,1)
        Kernel_model = KernelRegModel(self.df)
        SVI_model = SVIModel(self.df,self.S0)
        Polynomial_model = PolyModel(self.df)

        # Fit implied volatility using different methods
        self.fitted_DVF = DVF_model.fit_DVF()
        self.fitted_kernel = Kernel_model.fit_gauss_kernel_reg()
        self.fitted_SVI = SVI_model.SVI_fit()
        self.fitted_poly = Polynomial_model.poly_regress_optimal()

        self.df_dict = {"DVF":self.fitted_DVF, "Kernel":self.fitted_kernel, "SVI":self.fitted_SVI,"Polynomial":self.fitted_poly}

    def compute_fitted_metrics(self):
        """
        Method for computing metrics of fitted volatility of different curve-fitting techniques
        return:
            df_metrics: dataframe containing the performance of the fitting models
        """
        df_metrics = pd.DataFrame()
        dK = 1

        for method, df in self.df_dict.items():
            print(method)
            # Data prep and cleaning
            # Compute option price for fitted implied volatilities
            df_prices = self.compute_fitted_market_price(df)
            # Interpolate option prices to ensure constant dK
            df_interpolated = fittedMetrics.interpolate_df(df_prices)

            # Create empty dataframe for storing metrics
            df_metric = pd.DataFrame({"Strike":df_interpolated.Strike, "fittedMarketPrice":df_interpolated.fittedMarketPrice})

            # Compute risk neutral density
            # --------------------------------------------
            rnd = [fittedMetrics.FDM_approx_der2(i,df_metric,dK) for i in range(3,len(df_metric)-2)]
            df_rnd = pd.DataFrame({'Strike': df_metric.Strike.loc[3:len(df_metric)-3],'RND':rnd})
            area_under = fittedMetrics.area_under_RND(df_rnd)

            # Compute error in market price data 
            # --------------------------------------------
            market_errors = fittedMetrics.market_price_metrics(self.df, df_metric)
            abs_error = market_errors[0]
            rmse = market_errors[1]
            r2 = market_errors[2]

            # Compute vertical bull spread for arbitrage 
            # ------------------------------------------
            Qs = [fittedMetrics.vertical_bull_spread(i,df_metric,dK) for i in range(2,len(df_metric))]
            # Having computed the spread check if the spread is within the boundary 
            ### implement reduce function 
            value = 0
            for Q in Qs:
                if (0 <= Q <= round(np.exp(-self.r*self.T),3)): value += 1
            bull_error = (1 -(value/len(Qs))) * 100
            # print(f"Bull spread error: {bull_error}%")

            # Compute butterfly option payoff for arbitrage 
            # ----------------------------------------------
            butterfly_count = sum(map(lambda x : x <= 0, rnd))
            butterfly_error = round(((butterfly_count/len(rnd)) * 100),2)

            # Add all metrics to dataframe 
            current_df = pd.DataFrame({"Area under":area_under,"ABS mean error":abs_error,"RMSE":rmse,"R^2":r2,"Bull spread error (%)":bull_error,"Butterfly error (%)":butterfly_error},index = [method])
            df_metrics = pd.concat([df_metrics,current_df],axis = 0)

        df_metrics = df_metrics.transpose()
        print(df_metrics)

    def fitted_methods_plot(self):
        """
        Method for plotting the fitted volatility smiles of original and curve-fitting techniques
        """
        plt.figure(figsize = (15,10))

        plt.scatter(self.df['Strike'],self.df['Implied volatility'], color = 'cornflowerblue', linewidths= 0.5, label = "Market Implied Volatility")
        plt.axvline(self.S0, color = 'orange', ls = '--', label = "at-the-money")
        
        plt.plot(self.fitted_DVF['Strike'],self.fitted_DVF['fitted_imp_vols'],color = 'red', ls = '-', lw = 1.2, label = 'DVF')
        plt.plot(self.fitted_kernel['Strike'],self.fitted_kernel['fitted_imp_vols'], color = 'green', ls = '-' , lw = 1.2, label = "Kernel Regression")
        plt.plot(self.fitted_SVI['Strike'],self.fitted_SVI['fitted_imp_vols'], color = 'purple', ls = '-' , lw = 1.2, label = "SVI")
        plt.plot(self.fitted_poly['Strike'],self.fitted_poly['fitted_imp_vols'], color = 'brown', ls = '-' , lw = 1.2, label = "Polynomial Regression")

        plt.title(f"Fitting implied volatility for {self.ticker} with maturity {self.maturity_date}")
        plt.xlabel("Strike")
        plt.ylabel("Implied volatility")

        plt.grid()
        plt.legend()
        plt.show()



# Class for analytical metrics of fitness 
class fittedMetrics:

    @staticmethod
    def interpolate_df(df):
        """"
        Method to interpolate dataframe to allow consistent dK
        args:
            df: dataframe consisting of fitted market prices
        return:
            df_final: dataframe with interpolated fitted market prices 
        """
        interpolate_f =  interp1d(df.Strike, df.fittedMarketPrice, kind='cubic')
        
        x_new = np.arange(df.Strike.min(),df.Strike.max(),1)
        y_interpolated = interpolate_f(x_new)

        df_final = pd.DataFrame({"Strike":x_new,"fittedMarketPrice":y_interpolated})
        return(df_final)

    @staticmethod
    def FDM_approx_der2(i,df,dK):
        """
        Method for computing butterfly spread = RND using FDM
        args:
            i: index
            df: dataframe of fitted data
            dK: step size of strike price
        return:
            value: value of RND 
        """
        num = df.fittedMarketPrice.loc[i+1] - 2*(df.fittedMarketPrice.loc[i]) + df.fittedMarketPrice.loc[i-1]
        den = np.square(dK)
        value = num/den
        return(value)

    @staticmethod
    def vertical_bull_spread(i,df,dK):
        """
        Method for computing vertical bull spread 
        args:
            i: index
            df: dataframe of fitted data
            dK: step size of strike price
        return:
            Q: value of bull spread
        """
        num = df.fittedMarketPrice.loc[i-1] - df.fittedMarketPrice.loc[i]
        den = dK
        Q = num / den 
        return(Q)

    @staticmethod
    def area_under_RND(df):
        """
        Method to compute area under graph 
        args:
            df: dataframe with RND values
        return:
            area_under: area under the RND
        """
        area_under = auc(df.Strike,df.RND) 
        return(area_under)

    def market_price_metrics(df, df_fitted):
        """
        Method for mean absolute error, rmse and r2
        args:
            df: original volatility data from Newton-Raphson
            df_fitted: fitted volatility data computed using fitting techniques
        returns:
            errors: list of different errors: abs mean error, RMSE and R^2
        """
        df_original = pd.DataFrame({"Strike":df.Strike,"originalMarketPrice":df['Market price']})
        
        # finding common rows between two DataFrames
        df_common = df_original.merge(df_fitted, how = 'inner' ,on = ['Strike'],indicator=False)

        # Compute error 
        abs_error = round(mean_absolute_error(df_common.originalMarketPrice,df_common.fittedMarketPrice),2)
        rmse = round(np.sqrt(mean_squared_error(df_common.originalMarketPrice,df_common.fittedMarketPrice)),2)
        r2 = r2_score(df_common.originalMarketPrice,df_common.fittedMarketPrice)

        errors = [abs_error,rmse,r2]
        return(errors)


# Fit implied volatility using DVF
class DVFModel(): 
    def __init__(self,df,dK):
        self.df = df
        self.dK = dK

    def fit_DVF(self):
        """
        Method for fitting implied volatility using DVF
        args:
            self
        returns:
            df_fitted: dataframe with the fitted implied vol
        """
        df_fitted = self.df.copy()
        # Fit curve using regression and DVF
        df_fitted['K'] = df_fitted.apply(lambda x: x.Strike, axis = 1)
        df_fitted['K^2'] = df_fitted.apply(lambda x: x.Strike**2, axis = 1)
        df_fitted['T'] = df_fitted.apply(lambda x: x.TTM, axis = 1)
        df_fitted['T^2'] = df_fitted.apply(lambda x: x.TTM**2, axis = 1)
        df_fitted['KT'] = df_fitted.apply(lambda x: x.Strike * x.TTM, axis = 1)

        # Target: implied volatlity 
        y = df_fitted['Implied volatility']
        # Explanatory variables
        X = df_fitted[['K','K^2','T','T^2','KT']] 
        # Apply regression
        regr = linear_model.LinearRegression()
        regr.fit(X,y)

        fitted_imp_vols = regr.predict(X)
        df_fitted['fitted_imp_vols'] = fitted_imp_vols
        return(df_fitted)


# Fit implied volatility using Kernel Regression
class KernelRegModel():
    def __init__(self,df):
        self.df = df

    # Model using statsmodel
    def fit_gauss_kernel_reg(self):
        """
        Method for fitting implied vol using kernel regression (using statsmodel)
        args:
            self
        returns:
            df_kernel: dataframe with the fitted implied vol
        """
        #df_cleaned  = self.df.drop(self.df[self.df['impliedVolatility'] < 1e-4].index)
        df_cleaned = self.df.copy()
        df_cleaned.index = np.arange(0,len(df_cleaned))
    
        x = np.array(df_cleaned['Strike'])
        y = np.array(df_cleaned['Implied volatility'])

        kr = KernelReg(endog = y, exog = x, var_type= "c", bw = "cv_ls")

        # If no argument then data_predict == exog
        y_pred = (kr.fit())[0]
 
        df_kernel = pd.DataFrame({"TTM":df_cleaned.TTM[0],"Strike":x,"fitted_imp_vols":y_pred})

        return(df_kernel)

    # Model from scratch
    def fit_gauss_kernel_reg_custom(self):
        """
        Method for fitting implied vol using kernel regression (from scratch)
        args:
            self
        returns:
            df_kernel: dataframe with the fitted implied vol
        """
        b = 10
        df_cleaned = self.df.copy()
        x = np.array(df_cleaned['Strike'])
        y = np.array(df_cleaned['Implied volatility'])

        def kernel_gauss(gauss_input):
            return (1/np.sqrt(2*np.pi))*np.exp(-0.5*gauss_input**2)

        def predict_gauss(x,y,X):
            ks = [kernel_gauss((xi-X)/b) for xi in x]
            ws = [len(x) * (kernel/np.sum(ks)) for kernel in ks]
            dot_product = np.dot(ws, y)/len(x)
            return(dot_product)

        X = np.arange(x.min(),x.max(),0.1)
        y_preds = np.array([])

        for nm in X:
            y_pred = predict_gauss(x,y,nm)
            y_preds = np.append(y_preds,y_pred)

        df_kernel = pd.DataFrame({"TTM":self.df.TTM[0],"Strike":X,"fitted_imp_vols":y_preds})

        return(df_kernel)

# Fit implied volatility using SVI
class SVIModel():
    def __init__(self,df, S0):
        self.df = df 
        self.S0 = S0

    # Moneyness function 
    @staticmethod
    def moneyness(K,S0):
        """
        Method to compute log moneyness
        args:
            K: strike price
            S0: spot price
        return:
            value: log moneyness value
        """
        value = np.log(K / S0) 
        return(value)

    @staticmethod
    def tot_imp_variance(ttm,imp_vol):
        value = ttm * np.square(imp_vol)
        return(value)

    # Define g(x)
    #####################################################################################
    # raw SVI function
    @staticmethod 
    def SVI(parameters, x):
        # Assign raw SVI parameters 
        a, b, rho, m, sigma = parameters
        raw_svi_value = a + b * (rho * (x - m) + np.sqrt((x-m)**2 + np.square(sigma)))
        return(raw_svi_value)

    # SVI first derivative function
    @staticmethod
    def SVI_prime(b, rho, x, m, sigma):
        value = b * (rho + (   (x-m) / np.sqrt(np.square(x-m) + np.square(sigma))))
        return(value)

    # SVI second derivative function 
    @staticmethod
    def SVI_prime_2(b, sigma, x, m):
        value = (b*np.square(sigma)) / ((np.square(x-m) + np.square(sigma)) ** 1.5)
        return(value)

    # Risk neutral density g(x)
    @staticmethod
    def RND(parameters, x):
        # Assign raw SVI parameters 
        a, b, rho, m, sigma = parameters

        # Compute derivatives and SVI values
        w = SVIModel.SVI(parameters, x)
        w_prime = SVIModel.SVI_prime(b, rho, x, m, sigma)
        w_prime_2 = SVIModel.SVI_prime_2(b, sigma, x, m)

        # Compute rnd 
        rnd_value = np.square(1-(x*w_prime)/(2*w)) - ((np.square(w_prime)/4)*((1/w) + 0.25)) + (w_prime_2/2)
        return(rnd_value)

    def params_SVI(self,implied_var, x_market, RND_epsilon):
        """"
        Method to compute parameters for SVI mode
        args:
            implied_var: implied variance
            x_market: market price
            RND_epsilon: slack variable for butterfly/RND constraint
        returns:
            params: SVI optimized parameters
        """
        # Generate boundaries
        x_max, x_min = x_market.max(), x_market.min()
        var_max, var_min = implied_var.max(), implied_var.min()
        
        # Initial parameter guess
        a_initial = 0.5 * var_min
        b_initial = 0.1
        rho_initial = -0.5
        m_initial = 0.1
        sigma_initial = 0.1

        # Define initial parameters 
        initial_params = (a_initial, b_initial, rho_initial, m_initial, sigma_initial)

        # Define boundaries of parameters 
        a_bounds = (1e-5,var_max)
        b_bounds = (1e-3,1)
        rho_bounds = (-1,1)
        m_bounds = (2*x_min,2*x_max)
        sigma_bounds = (1e-2,1)

        lower_bounds = [a_bounds[0],b_bounds[0],rho_bounds[0],m_bounds[0],sigma_bounds[0]]
        upper_bounds = [a_bounds[1],b_bounds[1],rho_bounds[1],m_bounds[1],sigma_bounds[1]]
        bounds = Bounds(lower_bounds, upper_bounds)
            
        # Define butterfly constraint 
        butterfly_constraint = [{'type': 'ineq', 'fun': lambda x: SVIModel.RND(x, x_market) - RND_epsilon}]
        
        # Define objective function to be minimise 
        def objective_function(params, x, implied_var):
            # return vector norms of variance computed 
            result = np.linalg.norm(SVIModel.SVI(params, x) - implied_var, 2)
            return(result)

        # Preform optimisation of OLS
        res = minimize(
                    fun = lambda x : objective_function(x, x_market, implied_var), # Objective function to minimise
                    x0 = initial_params, # Initial guess of parameters
                    method='SLSQP', # Type of solver set to sequential least squares
                    bounds=bounds, # Bounds of variables
                    constraints=butterfly_constraint # Constraints for optimisation 
                    ) 

        # Solution array 
        params = res.x 
       
        return(params)

    def SVI_fit(self):
        """
        Method to fit volatility using SVI optimized parameters
        args:
            self
        returns:
            df_fitted: SVI fitted volatility
        """
        # Clean and prepare data
        df_var = pd.DataFrame()
        df_cleaned = self.df.copy()
        df_cleaned.index = np.arange(0,len(df_cleaned))
        TTM = df_cleaned.TTM[0]
        
        df_var['Moneyness'] = self.df.apply(lambda x: SVIModel.moneyness(x.Strike, self.S0),axis=1)
        df_var['impliedVarianceMarket'] = self.df.apply(lambda x: SVIModel.tot_imp_variance(TTM, x['Implied volatility']),axis=1)
        
        # Compute parameters - epsilon slack parameter added for constraint
        params = self.params_SVI(df_var.impliedVarianceMarket, df_var.Moneyness, 0.01)

        df_fitted = pd.DataFrame()
        df_fitted['Strike'] = self.df.Strike
        df_fitted['Moneyness'] = df_var.Moneyness
        df_fitted['fittedVariance'] = df_var.apply(lambda x: SVIModel.SVI(params, x.Moneyness),axis=1)
        # Convert total variance to volatility and moneyness to strike 
        df_fitted['fitted_imp_vols'] = df_fitted.apply(lambda x: np.sqrt(x.fittedVariance/TTM),axis = 1)
        
        return(df_fitted)

        
class PolyModel():
    def __init__(self,smile_data_df):
        self.smile_data_df = smile_data_df
        self.strikes = np.array(self.smile_data_df['Strike'])
        self.imp_vols = np.array(self.smile_data_df['Implied volatility'])
        
    @staticmethod
    def poly_regress_func(strikes_ar,imp_vols_ar,degree):
        strikes_ar_reshaped = strikes_ar.reshape(-1,1)
        # Select degree order for polynomial to be fitted 
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        # Set explanotry variable which is trike prices
        X_poly = polynomial_features.fit_transform(strikes_ar_reshaped)
        lin_reg = LinearRegression()
        model = lin_reg.fit(X_poly,imp_vols_ar)
        imp_vols_predicted = model.predict(X_poly)
        return(imp_vols_predicted)

    def paramter_tuning(self):
        """
        Method for hyperparameter tuning
        """
        # List for different degrees
        degrees = np.arange(2,3,1)

        # dictionary for metrics 
        # - key: degree 
        # - values: rmse,r2
        performance = dict()

        # Implement loop for iterating over different degrees
        for degree in degrees:
            # Compute regression 
            imp_vols_predicted = self.poly_regress_func(self.strikes,self.imp_vols,degree)
            # Compute metrics
            rmse = np.sqrt(mean_squared_error(self.imp_vols,imp_vols_predicted))
            r2 = r2_score(self.imp_vols,imp_vols_predicted)
            # Update performance metric dictionary
            performance[degree] = rmse, r2

        # Evaluate metrics
        df = pd.DataFrame(performance)
        df.index = ["RMSE","R2"]
        df = df.transpose()

        # Select performance with highest r^2 score
        optimal_R2 = df['R2'].max()
        degree_optimal = (df[df['R2']==optimal_R2].index).values

        self.degree_optimal = degree_optimal[0]

    def poly_regress_optimal(self):
        """
        Method for running hyperparameter tuning
        args:
            self
        returns:
            polf_df: fitted volatility from using polynomial regression
        """
        # Run parameter tuning method
        self.paramter_tuning()

        # Optimal predicted values
        imp_vols_predicted = self.poly_regress_func(self.strikes,self.imp_vols,self.degree_optimal)

        #Remove negative values
        imp_vols_predicted = np.where(imp_vols_predicted > 0, imp_vols_predicted,0)

        #Compute dataframe to be returned 
        poly_dict = {'Strike':self.strikes, 'fitted_imp_vols':imp_vols_predicted}
        poly_df = pd.DataFrame(poly_dict)
        
        return(poly_df)



    
    
   



