import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statistics import mode 
from collections import Counter
import warnings
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics

# define function to verify that variable is static for key
#------------------------------------------------------------------------------
def is_static(dframe, key, variable):
    # Function that checks whether "key" is associated with a unique value of "variable"   
    collapse = dframe.groupby(key)[variable].nunique() # returns number of unique "variable" for each "key"
    return str(variable) + ' is static for ' + str(key) + ': ' + str((collapse == 1).all())
#------------------------------------------------------------------------------    


# define function to verify that variable is static and return mode
#------------------------------------------------------------------------------
def pivot_static(x):
    # Function that catches inconsistencies when aggregating asserted static variables   
    if x.nunique() == 1: # returns number of unique "variable" for each "key"
        return mode(x)
    else:
        return np.nan # return nan if variable is not static for key in pivot_table
#------------------------------------------------------------------------------    

    
# define function to return number of days between first and last date in x 
#------------------------------------------------------------------------------
def daterange(x):
    return (max(pd.to_datetime(x))-min(pd.to_datetime(x))).days
#------------------------------------------------------------------------------


# define function to test goodness of fit with Benfords law
#------------------------------------------------------------------------------
def fraud_LR(x):
    x     = np.abs(x) #take absolute value to make sure F doesnt contain "-"
    F     = [int(str(i)[0]) for i in x] # construct list of first significant digits (FSDs)
    F     = sorted(Counter(list(filter(lambda dig: dig != 0, F))).most_common(9)) #filter zeros and count occurences .most_common(n) returns ordered tuples and sorted sorts lexicografically
    n     = sum(i for _ , i in F)   #get number of transactions       
    P_hat = [f/n for _ , f in F]  #get frequency of FSDs
    P     = [np.log10(1+1/d) for d in range(1,10)]  #calc. distribution fcn. of Benfords law
    LR    = 2*n*sum([p_hat*np.log(p_hat/p) for p_hat , p in list(zip(P_hat, P))]) # Calculate 2 ln Lambda .~ Chisq(8), where lambda is likelihood ratio
    return LR
#------------------------------------------------------------------------------


# define function to calculate and extract pay_delay trend for each user_id
#------------------------------------------------------------------------------
def pay_delay_trend(df, transaction_type):
    OUT = pd.DataFrame(columns = ('user_id', str(transaction_type) + '_delay_trend'))
    OUT['user_id'] = df['user_id'].unique()
    for user in OUT['user_id']: # Loop over unique user_id's
        try:
            # if-fork for the cases credit, debit and both (default) 
            if transaction_type == 'credit':
                delays = df[(df['user_id'] == user) & (df['remaining_transaction_amt'] == 0) & (df['transaction_amt'] < 0)][['trans_payment_due_date', 'pay_delay']].sort_values(by = 'trans_payment_due_date').dropna() # get due_dates and pay_delays for user 
            elif transaction_type == 'debit':
                delays = df[(df['user_id'] == user) & (df['remaining_transaction_amt'] == 0) & (df['transaction_amt'] > 0)][['trans_payment_due_date', 'pay_delay']].sort_values(by = 'trans_payment_due_date').dropna() # get due_dates and pay_delays for user 
            else:
                delays = df[(df['user_id'] == user) & (df['remaining_transaction_amt'] == 0)][['trans_payment_due_date', 'pay_delay']].sort_values(by = 'trans_payment_due_date').dropna() # get due_dates and pay_delays for user 
            delays['reltime'] = (pd.to_datetime(delays['trans_payment_due_date'])-np.min(pd.to_datetime(delays['trans_payment_due_date']))).astype('timedelta64[D]') # calculate number of days since first obs
            delays['reltime'] = delays['reltime']/np.max(delays['reltime']) # standardize time
            delays = delays.groupby('reltime')['pay_delay'].mean() # take mean of delays with the same time stamp
            delays = (delays-delays.mean())/delays.std() # standardize delays
            OUT.loc[OUT['user_id'] == user,str(transaction_type) + '_delay_trend'] = np.polyfit(delays.index, delays, 1)[0] # return the first order term. Note: [1] is the intercept!  
        except np.linalg.LinAlgError as e: # This code checks whether the thrown error is related to degenerate SVD
            if 'Singular matrix' in str(e):
                OUT.loc[OUT['user_id'] == user,str(transaction_type) + '_delay_trend'] = 0 # Define slope of point to be zero
        except:
            OUT.loc[OUT['user_id'] == user,str(transaction_type) + '_delay_trend'] = np.nan # return nan if e.g. no completed transactions
    return OUT
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
## Figures
#------------------------------------------------------------------------------ 

# define test function for fraud_LR, MonteCarlo distribution under H0 & H1 and compare to theoretical
#------------------------------------------------------------------------------
def test_fcn_fraud_LR(rep, sample_size):
    # allocate empty arrays
    test_dist1 = np.zeros(rep)
    test_dist0 = np.zeros(rep)
    
    # loop over rep and calculate test statistic for Normal (H1) and Log-Normal (H0)
    for r in range(rep):
        test_dist1[r] = fraud_LR(np.random.normal(0, scale=10, size=sample_size))
        test_dist0[r] = fraud_LR(np.exp(np.random.normal(0, scale=10, size=sample_size)))
    
    # produce figure to examine distribution of test statistic
    x = np.arange(0, 75, .2)
    fig = plt.figure()
    fig.suptitle('Distribution of test statistic', fontsize = 14)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('H0')
    ax2.title.set_text('H1')  
    ax1.hist(test_dist0, density=1, bins = int(rep/50), label='Empirical distn.')
    ax1.plot(x, stats.chi2.pdf(x, df=8), color='r', lw=2, label='Chi2 (8 d.f.)')
    ax1.legend(prop={'size': 10})
    ax2.hist(test_dist1, density=1, bins = int(rep/50))
    ax2.plot(x, stats.chi2.pdf(x, df=8), color='r', lw=2)
    ax1.set_xlim(0,70)
    ax2.set_xlim(0,70)
    ax2.set_yticks([])
    plt.show()
#------------------------------------------------------------------------------   

# define function to display fraud coef by bad
#------------------------------------------------------------------------------
def fraud_figure(df, min_transactions, alpha):
        
        crit = stats.chi2.ppf(1-alpha, df=8)

     # allocate lists
        bars1 = np.nanmean(df[(df[('transaction_amt', 'fraud_LR')] > crit) & (df[('num_transactions', 'sum')] > min_transactions)][('bad', 'pivot_static')]) # proportion of defaulted companies w. positive fraud test
        bars2 = np.nanmean(df[(df[('transaction_amt', 'fraud_LR')] < crit) & (df[('num_transactions', 'sum')] > min_transactions)][('bad', 'pivot_static')]) # proportion of defaulted companies w. negative fraud test
        l1 = len(df[(df[('transaction_amt', 'fraud_LR')] > crit) & (df[('num_transactions', 'sum')] > min_transactions)][('bad', 'pivot_static')]) # number of companies with positive fraud test
        l2 = len(df[(df[('transaction_amt', 'fraud_LR')] < crit) & (df[('num_transactions', 'sum')] > min_transactions)][('bad', 'pivot_static')]) # number of companies with negative fraus test
        # set width of bar
        barWidth = 0.35
        # Set position of bar on X axis
        r1 = 1
        r2 = 1.5
        # Make the plot
        plt.bar(r1, bars1, color='red', width=barWidth, edgecolor='white', label='Positive fraud test (' + str(l1) + ' companies)' )
        plt.bar(r2, bars2, color='green', width=barWidth, edgecolor='white', label='Negative fraud test (' + str(l2) + ' companies)' )
        plt.ylabel('Default rate', fontweight='bold',fontsize=12)
        plt.legend()
        plt.xticks([])
        plt.ylim((0, 0.05))
        plt.yticks(fontsize=12)
        plt.suptitle('Default rate by fraud test outcome', fontweight='bold', fontsize=14)
        plt.title('alpha:' + str(alpha) + ' & >' + str(min_transactions) + ' transactions')
        plt.show()
#------------------------------------------------------------------------------



# define function to display CF/capEX by segment and bad
#------------------------------------------------------------------------------  
def relative_CF_figure(df, by_segment):
    
    if by_segment == True:
        categories = sorted(df[('company_segment', 'pivot_static')].unique())    
        # allocate lists
        bars1 = []
        bars2 = []     
        # catch runtime warning thrown by np.nanmean on empty arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # set height of bar
            for i in range(len(categories)):
                bars1.append(np.log10(np.nanmean(df[(df[('company_segment', 'pivot_static')] == str(categories[i])) & (df[('bad', 'pivot_static')] == 0)]['cf/capex']))) # ratio for good companies
                bars2.append(np.log10(np.nanmean(df[(df[('company_segment', 'pivot_static')] == str(categories[i])) & (df[('bad', 'pivot_static')] == 1)]['cf/capex']))) # ratio for bad companies
           
            # set nan to 0 (this only changes segment = x, bad = 1)
            bars1 = [0 if np.isnan(x) else x for x in bars1]
            bars2 = [0 if np.isnan(x) else x for x in bars2]
    
        # set width of bar
        barWidth = 0.35
        # Set position of bar on X axis
        r1 = np.arange(len(bars1))
        r2 = [x + barWidth for x in r1]
        # Make the plot
        plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='Non-defaulted')
        plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='Defaulted')
        # Add xticks on the middle of the group bars and axis labels
        plt.xlabel('Company segment', fontweight='bold', fontsize=12)
        plt.ylabel('$\mathbf{Log_{10}}$ CF/Capex', fontweight='bold',fontsize=12)
        plt.ylim((0,8))
        plt.xticks([r + 0.5*barWidth for r in range(len(bars1))], categories, fontsize=12)
        plt.yticks(fontsize=12)
        # Create legend & Show graphic
        plt.legend()
        plt.title('Relative cashflow by company segment', fontweight='bold', fontsize=14)
        plt.show()
    else:
        # allocate lists
        bars1 = np.nanmean(df[df[('bad', 'pivot_static')] == 0]['cf/capex']) # ratio for good companies
        bars2 = np.nanmean(df[df[('bad', 'pivot_static')] == 1]['cf/capex']) # ratio for bad companies
        # set width of bar
        barWidth = 0.35
        # Set position of bar on X axis
        r1 = 1
        r2 = 1.5
        # Make the plot
        plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='Non-defaulted')
        plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='Defaulted')
        plt.ylabel('CF/Capex', fontweight='bold',fontsize=12)
        plt.legend()
        plt.xticks([])
        plt.yticks(fontsize=12)
        plt.title('Relative cashflow by default status', fontweight='bold', fontsize=14)
        plt.show()
#------------------------------------------------------------------------------

# define function to display PCL by bad
#------------------------------------------------------------------------------
def PCL_figure(df):
     # allocate lists
        bars1 = np.nanmean(df[df[('bad', 'pivot_static')] == 0][('PCL', 'mean')]) # PCL proportion for good companies
        bars2 = np.nanmean(df[df[('bad', 'pivot_static')] == 1][('PCL', 'mean')]) # PCL proportion for bad companies
        # set width of bar
        barWidth = 0.35
        # Set position of bar on X axis
        r1 = 1
        r2 = 1.5
        # Make the plot
        plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='Non-defaulted')
        plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='Defaulted')
        plt.ylabel('Proportion PCL transactions', fontweight='bold',fontsize=12)
        plt.legend()
        plt.xticks([])
        plt.yticks(fontsize=12)
        plt.title('PCL by default status', fontweight='bold', fontsize=14)
        plt.show()
    


#------------------------------------------------------------------------------ 

# define function to display pay_delay trend by bad and transaction type
#------------------------------------------------------------------------------
def pay_delay_figure(df):       
    # allocate lists 
    bars1 = []
    bars2 = []     
    categories = ('credit', 'debit', 'all')
    # set height of bar
    for transaction_type in categories:
        bars1.append(np.mean(df[df[('bad', 'pivot_static')] == 0][str(transaction_type) + '_delay_trend'])) # average trend for good companies
        bars2.append(np.mean(df[df[('bad', 'pivot_static')] == 1][str(transaction_type) + '_delay_trend'])) # average trend for bad companies
    # set width of bar
    barWidth = 0.35
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='Non-defaulted')
    plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='Defaulted')
    # Add xticks on the middle of the group bars and axis labels
    plt.xlabel('Transaction type', fontweight='bold', fontsize=12)
    plt.ylabel('Trend, standardized delay', fontweight='bold',fontsize=12)
    plt.xticks([r + 0.5*barWidth for r in range(len(bars1))], categories, fontsize=12)
    plt.yticks(fontsize=12)
    # Create legend & Show graphic
    plt.legend()
    plt.title('Trans. lateness trend by trans. type', fontweight='bold', fontsize=14)
    plt.show()
    
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
## Modelling and prediction
#------------------------------------------------------------------------------ 


# define wrapper function for Logistic regression
#------------------------------------------------------------------------------
def logreg_train(X_train, Y_train, X_test, Y_test):
    
    # Create logistic regression model object
    classifier = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                                            fit_intercept=True, intercept_scaling=1, class_weight=None, 
                                            random_state=None, solver='lbfgs', max_iter=1000, multi_class='auto', 
                                            verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)    
    # Fit classifier using the training data
    classifier.fit(X_train, Y_train)
    return classifier
#------------------------------------------------------------------------------


# define function to evaluate sklearn object
#------------------------------------------------------------------------------
def eval_classifier(classifier, X_train, Y_train, X_test, Y_test, cutoff):
    # Evaluate on training data
    print('\n-- Training data --')
    decisions = (classifier.predict_proba(X_train) >= cutoff)[:,1].astype(int)
    accuracy = sklearn.metrics.accuracy_score(Y_train, decisions)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_train, decisions))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_train, decisions))
    print('')
    # Evaluate on test data
    print('\n---- Test data ----')
    decisions = (classifier.predict_proba(X_test) >= cutoff)[:,1].astype(int)
    accuracy = sklearn.metrics.accuracy_score(Y_test, decisions)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_test, decisions))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_test, decisions))
    print('Coefficients:')
    coefs = pd.DataFrame(data = classifier.coef_, columns = list(X_train.columns))
    print(coefs)
#------------------------------------------------------------------------------











