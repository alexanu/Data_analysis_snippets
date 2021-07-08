from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy.stats as scs
import pandas as pd


# BOOTSTRAP (averaging estimates from multiple small data samples) ######
    '''
    The bootstrap method is used to estimating quantities about a population ...
    ... by averaging estimates from multiple small data samples.
    Samples are constructed by drawing observations from a large data sample one at a time ...
    ... and returning them to the data sample after they have been chosen.
    This allows a given observation to be included in a given small sample more than once.
    This approach to sampling is called sampling with replacement.

    A quantity of a population is estimated by repeatedly taking small samples,
    calculating the statistic, and taking the average of the calculated statistics.
    '''

    from sklearn.utils import resample
    data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # data sample
    # prepare bootstrap sample
    # with replacement means values are returned back
    boot = resample(data, replace=True, n_samples=4, random_state=1)
    oob = [x for x in data if x not in boot]  # out of bag observations
    print('Bootstrap Sample: %s' % boot)
    print('OOB Sample: %s' % oob)

    %matplotlib inline
    data = pd.read_csv('pima-indians-diabetes.csv', header=None)  # load dataset
    values = data.values
    # configure bootstrap
    n_iterations = 1000
    n_size = int(len(data) * 0.50)
    # run bootstrap
    stats = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(values, n_samples=n_size)
        test = np.array(
            [x for x in values if x.tolist() not in train.tolist()])
        # fit model
        model = DecisionTreeClassifier()
        model.fit(train[:, :-1], train[:, -1])
        # evaluate model
        predictions = model.predict(test[:, :-1])
        score = accuracy_score(test[:, -1], predictions)
        stats.append(score)
    # plot scores
    pyplot.hist(stats)
    pyplot.show()
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' %
            (alpha*100, lower*100, upper*100))


# Nonparametric Rank Correlation (Spearman’s rho, Kendall’s tau) #####

    '''
    When we do not know the distribution of the variables (i.e. not Gaussian), ...
    ... we must use nonparametric rank correlation methods.
    This is done by first converting the values for each variable into rank data.
    Rank correlation methods are referred to as distribution-free correlation or nonparametric correlation
    Examples: Spearman’s Rank Correlation.
            Kendall’s Rank Correlation.
            Goodman and Kruskal’s Rank Correlation.
            Somers’ Rank Correlation
            
        Spearman’s rank correlation (aka Spearman’s rho):
            degree to which ranked variables are associated by a monotonic function
            calculates a Pearson’s correlation (e.g. a parametric measure of correlation) ...
            ... using the rank values instead of the real values
                Pearson’s correlation is the calculation of the covariance 
                    (or expected difference of observations from the mean) 
                    between the two variables normalized by the variance or spread of both variables.

        Kendall’s rank correlation (aka Kendall’s tau):
            calculates a normalized score for the number of matching ...
            ... or concordant rankings between the two samples

    '''

        # each variable is drawn from a uniform distribution (e.g. non-Gaussian) ...
        # ...and the values of the second variable depend on the values of the first value
        from numpy.random import rand
        from numpy.random import seed
        from matplotlib import pyplot
        %matplotlib inline
        seed(1)
        data1 = rand(1000) * 20
        data2 = data1 + (rand(1000) * 10)
        pyplot.scatter(data1, data2)
        pyplot.show()

        # calculate spearman's correlation
            from scipy.stats import spearmanr
            coef, p = spearmanr(data1, data2)
            print('Spearmans correlation coefficient: %.3f' % coef)
            alpha = 0.05
            if p > alpha:
                print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
            else:
                print('Samples are correlated (reject H0) p=%.3f' % p) 

        # calculate kendall's correlation
            from scipy.stats import kendalltau
            coef, p = kendalltau(data1, data2)
            print('Kendall correlation coefficient: %.3f' % coef)
            alpha = 0.05
            if p > alpha:
                print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
            else:
                print('Samples are correlated (reject H0) p=%.3f' % p)


# Chi square function which performs an independence test btw 2 cols of a data frame

    def chi_square_of_df_cols(df, col1_name, col2_name):
        df_col1, df_col2 = df[col1_name], df[col2_name]
        result = []
        # In the outer loop we go over every category for the first variable
        for cat1 in categories(df_col1):
            cat1_value_list = []
            # now we loop over every category in the second variable
            for cat2 in categories(df_col2):
                # number of occurences
                num_measurements = len(df[(df_col1 == cat1) & (df_col2 == cat2)])
                cat1_value_list.append(num_measurements)
            result.append(cat1_value_list)
        chi2, p_value, df, expected = scs.chi2_contingency(
            result)  # scipy stats run the chi square test
        return chi2, p_value, df  # chi2, p value and degrees of freedom

    # Lets perform the chi2 test for every collum against every other
    interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age', 'Involvement']
    n = len(interesting_collums)
    pval_mat = np.empty((n, n))
    for i, name1 in enumerate(interesting_collums):
        # for every collum go through all other collums
        for j, name2 in enumerate(interesting_collums):
            chi2, p_value, df = chi_square_of_df_cols(data, name1, name2)  # perform the chi2 of those two
            pval_mat[i, j] = p_value

    # Save all the p vals as a text file which you can load e.g. in excel
    pval_df = pd.DataFrame(pval_mat, columns=interesting_collums, index=interesting_collums)
    pval_df.to_excel('/Users/.../data/all_chi2_pvals.xls')
    np.savetxt('/Users/.../data/chi2_pvals.csv', pval_mat)


# Kolmog-Smirnov to compare distributions
    from scipy.stats import ks_2samp
    def run_ks_test(df1, df2):
        """ Run Kolmogorov-Smirnov test for each column
        H0: both follow the same distribution """
        if df1.shape[1] != df2.shape[1]:
            raise ValueError('Shape mismatch. df1={}, df2={}'.format(df1.shape, df2.shape))

        ks_test_result = []
        for col_name in df1:
            ks_stat, p_value = ks_2samp(df1[col_name], df2[col_name])
            ks_test_result.append([col_name, ks_stat, p_value])
        ks_test_result = pd.DataFrame(ks_test_result, columns=['col_name', 'ks_test', 'p_value'])
        return ks_test_result