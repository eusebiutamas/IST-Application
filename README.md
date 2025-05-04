# This is the Intraday Stock Trading (IST) Application 

##

This project recommends an intraday stock trading strategy by using three basic deep learning algorithms such as the  Multi
Layer Perception (MLP), Convolutional Neural Network (CNN), Long Short-
Term Memory (LSTM), and additionally a hybrid CNN-Bi-LSTM optimized with Genetic Algorithm (GA). 
The data used in this project is the NASDAQ-100 dataset exracted from the Agoseek website (URL source: https://www.algoseek.com/ml4t-book-data.html).
To use the same dataset, simply click on the "Download algoseek data used in ML4T" and then move it to the data folder within this project.

## Install and run the program:

### 1. Install Python 3.11

Download and install Python 3.11 using the following link:
https://www.python.org/downloads/release/python-3110/
Navigate to Files and choose you desired file based on your environment (e.g. Windows, Linux, Mac)

Make sure to select the option **'Add Python to PATH'** throughout the installation process

### 2. Create a virtual environment:

Open a command prompt and navigate to the file of your newly created project and run the following:
```bash
python -m venv .venv
```
### 3. Activate the environment:
.venv\Scripts\activate
### 4. Intall the requirements:
pip intall -r requirements.txt
### 4. Run the project:
- **Firstly run the main file GUI:** python main.py
- **Secondly run the file:** python vectorized_backtest.py
- **Thirdly run the GUI:** systemUI.py

By running the main.py file, the full system of the deep learning model's pipeline can be tested from the first step of data pre-processing, moving to the feature engineering and then train, test, and save predictions for backtesting purposes.

Running the vectorized_backtest.py, a simple intraday trading strategy on the price movements using the models' prediction can be illustrated.

Within the systemUI.py file the prediction can be backtested by clicking the predict button.
Other functionalities can be used, such as searching for a desired ticker from all 100 tickers within NASDAQ-100 dataset, change the plot style, select a desired time frame from 1-min candlesticks to 1-hour, zoom in and out, and pause and resume the simulation.

### References
Jansen, S., 2020. *Machine Learning for Algorithmic Trading: Predictive models to extract signals from market and alternative data for systematic trading strategies with Python*. Packt Publishing Ltd.

Wirsansky, E., 2020. *Hands-On Genetic Algorithms with Python: Applying genetic algorithms to solve real-world deep learning and artificial intelligence problems*. Packt Publishing Ltd.
