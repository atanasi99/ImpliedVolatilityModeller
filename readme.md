# implied volatility modeller

***

### Table of contents

- Introduction

- Requirements

- Installation

- Configuration

- Troubleshooting 

- Support and issues

***

### Introduction

The "implied volatility modeller" is a program written in Python with the objective of modelling implied volatility using a computational and mathematical approach. 

### Requirements

The program is written in Python and requires the local machine running the program to be using Python 3.6.8 or later. The code can be run from the terminal or run from an IDE. Configuration and running the code will be explained in section 4. 

### Installation

For the recommended modules used in the program please refer to using the requirements.txt file. Using the requirements.txt file, pip installs all the modules in the file. To do this first open your terminal then change your current directory to the directory where the requirments.txt file is and follow the command

```
pip install -r requirements.txt
```

### Configuration

The program requires 8 Python files to run:

- main.py - is the file that executes the program
- analytics.py - is the file that contains the code for volatility analytics including volatility smile and surface computation.

- bs_library.py - is the file containing the Black-Scholes library 
- deribit_market_data.py - is the file containing the class and methods for retrieving crypto data from Deribit exchange
- fitted_vol_analytics.py - is the file containing the classes of each curve-fitting technique and methods for fitting the implied volatility 
- market_data.py - is the file containing class and methods for retrieving market data from Yahoo Finance and Bank of England
- Newton_Raphson_Method.py - is the file containing the Newton algorithm for computing the implied volatility 

For executing the program only one file will have to be executed which is the main.py. Using the terminal change directory to the location of the files then run:

```
python .\main.py
```

Using this command will execute the program and allow the user to interact in the terminal. Other option is to open the program folder in an IDE and run the main.py file from there.

On the main menu:

- entering "1" - provide a computation of implied volatility for specified option either manual configuration or using market data

- entering "2" - provide computation of volatility smile
- entering "3" - provide computation of volatility surface
- entering "4" - provide computation of fitted volatility analytics
- entering "5" - enter option portfolio (this requires having MySQL setup on device)
- entering "6" - exit program 

### Troubleshooting

The code focuses on mostly research, hence requires ensuring that inputs are of the correct data type when entered.  During the execution of the program pressing "ctrl+c" will bring the program to the main menu and entering "6" will exit the program. If any mistakes are made to the input just follow "ctrl+c" and then select any of the functionalities.

### Support and issues

For any queries regarding running the program or any other FAQ please email at:

**K21108925@kcl.ac.uk**



