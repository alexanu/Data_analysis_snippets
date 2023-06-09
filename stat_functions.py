import pandas as pd
import matplotlib.pyplot as plt



# Linear Regression
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    def linear_regression(basis_X_train, basis_y_train, basis_X_test,basis_y_test):
        regr = linear_model.LinearRegression()
        regr.fit(basis_X_train, basis_y_train) # Train the model using the training sets
        basis_y_pred = regr.predict(basis_X_test) # Make predictions using the testing set

        print('Coefficients: \n', regr.coef_)
        print("Mean squared error: %.2f" % mean_squared_error(basis_y_test, basis_y_pred))
        print('Variance score: %.2f' % r2_score(basis_y_test, basis_y_pred)) # Explained variance score: 1 is perfect prediction

        plt.scatter(basis_y_pred, basis_y_test,  color='black')
        plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)
        plt.xlabel('Y(actual)')
        plt.ylabel('Y(Predicted)')
        plt.show()
        
        return regr, basis_y_pred


# Augmented Dickey Fuller test

    from statsmodels.tsa.stattools import adfuller

    def adf_test(series):
        '''
        Function for running the Augmented Dickey Fuller test and displaying the results in a human-readable format. The number of lags is chosen automatically based on AIC.
        
        Null Hypothesis: time series is not stationary
        Alternate Hypothesis: time series is stationary

        Args:
        * series - a pd.Series object containing the time series to investigate
        '''
        adf_test = adfuller(series, autolag='AIC')
        adf_results = pd.Series(adf_test[0:4], index=['Test Statistic',
                                                    'p-value',
                                                    '# of Lags Used',
                                                    '# of Observations Used'])
        for key, value in adf_test[4].items():
            adf_results[f'Critical Value ({key})'] = value

        print('Results of Augmented Dickey-Fuller Test ----')
        print(adf_results)

# kpss 
    from statsmodels.tsa.stattools import kpss

    def kpss_test(series, h0_type='c'):
        '''
        Function for running the KPSS test and displaying the results in a human-readable format.

        Null Hypothesis: time series is stationary
        Alternate Hypothesis: time series is not stationary

        When null='c' then we are testing level stationary, when 'ct' trend stationary.

        Args:
        * series - a pd.Series object containing the time series to investigate
        * h0_type - string, what kind of null hypothesis is tested
        '''
        kpss_test = kpss(series, regression=h0_type)
        kpss_results = pd.Series(kpss_test[0:3], index=['Test Statistic',
                                                        'p-value',
                                                        '# of Lags'])
        for key, value in kpss_test[3].items():
            kpss_results[f'Critical Value ({key})'] = value

        print('Results of KPSS Test ----')
        print(kpss_results)

# Autocorrelation

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    def test_autocorrelation(series, h0_type='c'):

        fig, ax = plt.subplots(2, figsize=(16, 8))
        plot_acf(series, ax=ax[0], lags=40, alpha=0.05)
        plot_pacf(series, ax=ax[1], lags=40, alpha=0.05)
        
        adf_test(series)
        kpss_test(series, h0_type='c')
        print('Autocorrelation plots ----')
        plt.show()


# MAPE, MAE, RMSE

    # Evaluating accuracy of time series model with error metrics 
    # Use these metrics to evaluate our prediction accuracy

    # MAPE - mean absolute percentage error
    def get_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # MAE - mean absolute error
    def get_mae(x, y):
        return np.mean(abs(np.array(x)-np.array(y)))

    # RMSE - root mean squared error
    def get_rmse(x, y):
        return math.sqrt(np.mean((np.array(x)-np.array(y))**2))