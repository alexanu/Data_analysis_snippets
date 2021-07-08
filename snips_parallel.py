# Parallel programming with Pool

# Importing the multiprocessing 
from multiprocessing import Pool

# function to which we'll perform multi-processing
def cube(i):
    i = i+1
    z = i**3
    return z

# using pool class to map the function with iterable arguments-
print(Pool().map(cube, [1, 2, 3]))


#----------------------------------------------------------------

''' use threading module for paralel running of some function '''

import time
from threading import Thread

def no_arg(func, instances): # func is function withOUT arguments
    for i in range(instances): # number of threads equals instances
        t = Thread(target=func)
        t.start()

def with_arg(func, instances,args): # func is function with arguments
    for i in range(instances): # number of threads equals instances
        t = Thread(target=func, args = args) # arguments in tuple
	t.start()


#-----------------------------------------------------------------

def setup_parallel(tickers, mktdata='combined', n_process=3, 
                    baseDir = 'D:\\Financial Data\\Netfonds\\DailyTickDataPull', supress='yes'):
           
    #break up problem into parts (number of processes)
    length = len(tickers)
    index=[]
    df_list=[]
    for i in range(n_process):
        index.append(range(i,length, n_process)) 
        df = tickers.loc[index[i]] 
        df.index=range(len(df))
        df_list.append(df)
    
    queue = multiprocessing.Queue()
    
    #start the pull data processes
    jobs=[]
    for tickers in df_list:
        p = multiprocessing.Process(target=pull_tickdata_parallel, 
                                    args=(queue, tickers,latest_dates_df, 'combined', length, start, directory, supress))
        jobs.append(p)
        p.start()
    
    for j in jobs:
        j.join()
        
    print 'Joined other threads'


def pull_tickdata_parallel(queue, tickers, latest_date, mktdata='combined',nTot=0,sTime=0, directory='', supress='yes'):

    pName = multiprocessing.current_process().name    
    
    for i in tickers.index:
        name = tickers['ticker'][i]
        folder=tickers['folder'][i]
        data = multi_intraday_pull2(name, pd.datetime.date(start_date), date.date(), 30,mktdata, folder, directory)
          
        tempstr = '%-12s: %-10s: Iter=%5d'%(pName,name,i)+ ', %-3s'%data
        to_pass = ({name:date}, tempstr)
        queue.put(to_pass)  
    return 

#-----------------------------------------------------------------


