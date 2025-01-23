import pandas as pd
import time


# read txt with "with" and open()
    # filename = "myfile.txt"
    with open(filename, "r") as f: # automaticall close the file in the end
        for line in f:
            print(f)

        # This above is equivalent to this:
        filename = "myfile.txt"
        try: 
            f = open(filename, "r")
            for line in f:
                print(f)
        except Exception as e:
            raise e
        finally:
            f.close()


    with open('foo.txt', 'rt') as file: # 'rt' - open for reading (text)
        data = file.read() # `data` is a string with all the text in `foo.txt`

    with open(filename, 'rt') as file:
        for line in file:
            if line == '\n':    # Skip blank lines
                continue
            # More statements
            ...

    with open(filename) as f:
        for lineno, line in enumerate(f, start=1):
        ...

    filename = "menu.py"
    with open(filename) as in_file:
        text = "\n".join([line.rstrip() for line in in_file.readlines()]) + "\n"
    


    [line.strip() for line in open("readFile.py")] # here Python closes the file automatically




    with open('outfile', 'wt') as out:  # 'wt' - open for writing (text)
        out.write('Hello World\n')

    with open('outfile', 'wt') as out:
        print('Hello World', file=out)
        ...

    records = []  # Initial empty list

    with open('Data/portfolio.csv', 'rt') as f:
        next(f) # Skip header
        for line in f:
            row = line.split(',')
            records.append((row[0], int(row[1]), float(row[2])))

    import csv
    def portfolio_cost(filename):
        '''Computes the total cost (shares*price) of a portfolio file'''
        total_cost = 0.0

        with open(filename, 'rt') as f:
            rows = csv.reader(f)
            headers = next(rows)
            for row in rows:
                nshares = int(row[1])
                price = float(row[2])
                total_cost += nshares * price
        return total_cost


# load the tickers from the txt ticker file
    ticker_list = [x.strip() for x in open("tickers.txt", "r").readlines()]
    print("Number of equities: ", len(ticker_list))

    Tickers=[]
    ETF_directory = 'C:\\Users\\oanuf\\Data\\minute_data\\US\\ETF\\'
    Tickers.append([x.split('.')[0] for x in os.listdir(ETF_directory) if x.endswith(".txt")])


# Simple excel import
    import datetime as dt
    dateparse = lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    hist_earn_calls=pd.read_excel('D:\\Data\\Other_data\\all_5Y_earn_calls.xlsx',
                                    usecols=['RIC','Ticker', 'ISIN', 'Event_Type', 'Event_date_GMT'],
                                    parse_dates=['Event_date_GMT'], date_parser=dateparse)


# Create list of avialable files
    import os, glob
    import pandas as pd
    file_list = glob.glob(Minute_data_directory + '*.zip') # all files in directory
    hist_data = pd.DataFrame(file_list)
    hist_data['curr_pair'] = hist_data[0].apply(lambda x: x[10:16])
    hist_data['year'] = hist_data[0].apply(lambda x: x[-8:-4])


# Create overview of content of available csv files with quotes and  gesamt database
    import os
    import pandas as pd
    import time
    import datetime as dt

    Tickers=[]
    Tickers.append([x.split('_')[0] for x in os.listdir(Alpaca_directory) if x.endswith(".csv")])
    data_summary_df = pd.DataFrame(columns = ['symbol','start','end','num_rows'])
    alpaca_quotes=pd.DataFrame()
    for idx,ticker in enumerate(Tickers[0]):
        nameoffile=Alpaca_directory+ticker+"_ET_adj_alpaca.csv"
        data = pd.read_csv(nameoffile,index_col='timestamp', parse_dates=['timestamp'])
        data_summary_df.loc[idx+1] = [ticker, data.index[0], data.index[-1],len(data)]
        data = data.assign(ticker=ticker)
        alpaca_quotes=alpaca_quotes.append(data)
        print("Done {} and still {} to go".format(idx,len(Tickers[0])-idx))
    data_summary_df.to_excel("Alpaca_minute_quotes_overview.xlsx")
    alpaca_quotes.to_csv(Alpaca_directory+"Alpaca_min_quotes_ET_adj.csv")


# Read ETFDB
    TOP_US_TICKERS=[]
    hist_index_member=pd.read_excel(ETFDB_path,sheet_name='Stock_tickers',skiprows=0,header=1,usecols=['symbol', 'was_us_index_const','marketCap_Bn','sector']) # read hist index components
    hist_index_member = hist_index_member.dropna(axis = "rows") # drop any row that has missing values
    hist_index_member = hist_index_member[hist_index_member.was_us_index_const=="Yes"]
    [TOP_US_TICKERS.extend(hist_index_member[hist_index_member.sector==i].sort_values(['marketCap_Bn']).symbol[0:10].to_list()) for i in hist_index_member.sector.unique()]


# Read 1st and last row of txt ETF minute data + dates reading ----------------------

    import os.path, time
    import datetime as dt

    def readlastline(f):
        f.seek(-2, 2)              # Jump to the second last byte.
        while f.read(1) != b"\n":  # Until EOL is found ...
            f.seek(-2, 1)          # ... jump back, over the read byte plus one more.
        return f.read()            # Read all data from this point on.

    Tickers=[]
    ETF_directory = 'D:\\Data\\minute_data\\US\\Stocks_adj\\'
    Tickers.append([x.split('.')[0] for x in os.listdir(ETF_directory) if x.endswith(".txt")])
    ETF_data=[]
    for ticker in Tickers[0]:
        with open(ETF_directory+ticker+'.txt', "rb") as f:
            next(f) # this file has header => we skip it
            first_row = str(f.readline()).split(',')[0] # comma as separator
            last_row = str(readlastline(f)).split(',')[0] # comma as separator
        ETF_data.append([ticker, first_row, last_row]) # we read 1 ticker so we add data to the list
    ETF_minute =pd.DataFrame(ETF_data,columns = ['Ticker','Start_New','End_New']) # transform list of lists into panda
    ETF_minute['Start_New'] = ETF_minute['Start_New'].str.split("'").str[1] # there is "b" in the beginning, which is byte type, 
                                                                            # ... I didn't manage to get rid of it with .decode('utf-8')
    ETF_minute['End_New'] = ETF_minute['End_New'].str.split("'").str[1]

    ETF_minute['Start_New'] = pd.to_datetime(ETF_minute['Start_New'])
    ETF_minute['End_New'] = pd.to_datetime(ETF_minute['End_New'])
    ETF_minute['Start_End'] = ETF_minute['Start_New'].dt.strftime('%Y-%b') +" to "+ETF_minute['End_New'].dt.strftime('%Y-%b')
    ETF_minute.to_csv('ETF_with_minut.csv')


# Copy files which are not there yet -----------------------------------------
    import shutil
    ETF_directory_old = 'D:\\Data\\minute_data\\US\\ETF\\'
    ETF_directory = 'D:\\Data\\minute_data\\US\\'
    Old_Tickers=[]
    Old_Tickers.append([x.split('.')[0] for x in os.listdir(ETF_directory_old) if x.endswith(".txt")])
    New_Tickers=[]
    New_Tickers.append([x.split('.')[0] for x in os.listdir(ETF_directory) if x.endswith(".txt")])
    
    for ticker in list(set(Old_Tickers[0])-set(New_Tickers[0])):
        shutil.copy2(ETF_directory_old+ticker+'.txt', ETF_directory)


# Download many csvs --------------------------------------------------

    url = 'https://www.sectorspdr.com/sectorspdr/IDCO.Client.Spdrs.Portfolio/Export/ExportCsv?symbol='
    tickers = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    SPDRs = pd.concat((pd.read_csv(url+ticker, skiprows=1).assign(Ticker=ticker) for ticker in tickers), ignore_index=True)

    SPDRs = pd.concat((pd.read_excel(url+ticker+".xlsx", skiprows=5).assign(Ticker=ticker) for ticker in SPDR_ETF if requests.head(url+ticker+".xlsx").status_code == 200), ignore_index=True)


# ISO ESG --------------------------------------------------------------------------

    import pandas as pd

    CountryISO = ['LUX','USA','NOR','CHE','DNK','ISL','AUT','CAN','IRL','BEL','AUS','HKG','NLD',
                'JPN','GBR','DEU','FRA','FIN','MCO','SWE','ITA','LIE','SGP','TWN','ARE','ESP',
                'NZL','QAT','GRC','ISR','CYP','KWT','SVN','PRT','KOR','MLT','BHR','CZE','BRB',
                'HUN','SVK','OMN','URY','EST','SAU','LTU','MUS','ARG','POL','ZAF','HRV','LVA',
                'CHL','TTO','CRI','MEX','MYS','BWA','RUS','BRA','BGR','THA','NAM','IRN','ROU',
                'TUN','TUR','COL','KAZ','PAN','BLR','DZA','DOM','TKM','UKR','PER','CHN','BLZ',
                'VEN','LBN','PRY','PHL','ALB','JOR','GTM','EGY','MAR','JAM','LKA','ARM','AZE',
                'ECU','IDN','IND','CUB','HND','VNM','GEO','BOL','NIC','GHA','SRB','PNG','PAK',
                'CMR','MDA','MNG','UZB','KGZ','KEN','TJK','NGA','ATA','ASM','AND','ATG','BHS',
                'BMU','BTN','BVT','IOT','SLB','VGB','BRN','MMR','BDI','CPV','CYM','CAF','CXR',
                'CCK','COM','MYT','COG','COD','COK','BEN','DMA','SLV','GNQ','ERI','FRO','FLK',
                'SGS','FJI','ALA','GUF','PYF','ATF','DJI','GAB','GMB','PSE','GIB','KIR','GRL',
                'GRD','GLP','GUM','GIN','GUY','HTI','HMD','VAT','CIV','PRK','LSO','MAC','MDG',
                'MWI','MDV','MLI','MTQ','MRT','MNE','MSR','NRU','NPL','CUW','ABW','SXM','BES',
                'NCL','VUT','NIU','NFK','MNP','UMI','FSM','MHL','PLW','PCN','GNB','TLS','REU',
                'BLM','SHN','KNA','AIA','LCA','MAF','SPM','VCT','SMR','STP','SYC','SLE','SSD',
                'ESH','SUR','SJM','SWZ','TGO','TKL','TON','TCA','TUV','MKD','GGY','JEY','IMN',
                'VIR','WLF','WSM','PRI','LBY','BIH','SYR','BGD','SDN','KHM','ZWE','AGO','LAO',
                'SEN','IRQ','UGA','RWA','MOZ','TCD','BFA','LBR','YEM','NER','ZMB','ETH','AFG','TZA','SOM']
    Gradus=['0.5','1','1.5','2','2.5','3']
    Database_climate=pd.DataFrame()
    for country in CountryISO:
        for grad in Gradus:
            try:
                data = pd.read_csv('https://cie-api.climateanalytics.org/api/map-difference/?iso='+country+'&var=ec1&season=annual&format=csv&scenarios=cat_current&wlvls='+grad, skiprows=10)
                data.rename(columns={'lat \ lon': 'lat'}, inplace=True)
                data = data.melt(id_vars=["lat"], var_name="lon", value_name="Value") # put columns to row
                data = data.dropna() # cleaning
                data = data[data.Value != 0] # cleaning
                data = data.assign(Country=country,Grade=grad) # adding 2 columns
                Database_climate=Database_climate.append(data,ignore_index=True)
            except:
                print("no data for ",country," with grade ", grad)
                pass
    Database_climate.to_csv("Database_climate.csv")


# Get info from different places in line
    for line in listdir:
        line_chunks = line.split("_")
        year = line[10:16]
        name = line_chunks[0]
        gender = line_chunks[1]
        count = line_chunks[2]

        data_list.append([year, name, gender, count])


# Transform different txt-s to csv ----------------------------------------------------------------------------------------------
    stocks_directory = 'D:\\Data\\minute_data\\US\\ETF\\'
    new_stocks_directory = 'D:\\Data\\minute_data\\US\\'

    Tickers=[]
    Tickers.append([x.split('.')[0] for x in os.listdir(stocks_directory) if x.endswith(".txt")])
    for ticker in Tickers[0]:
        if dt.datetime.fromtimestamp(os.path.getmtime(stocks_directory+ticker+'.txt')).year==2013: # when the file was modified
            dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')
            read_file = pd.read_csv(stocks_directory+ticker+'.txt', sep='\t', decimal=",",parse_dates={'datetime': ['Date', 'Time']}, date_parser=dateparse,index_col=0)
            read_file=read_file.drop('Unnamed: 7',axis=1) # delete not needed columns
            read_file.to_csv(new_stocks_directory+ticker+'.txt')
        elif dt.datetime.fromtimestamp(os.path.getmtime(stocks_directory+ticker+'.txt')).year==2019:
            read_file = pd.read_csv(stocks_directory+ticker+'.txt', sep=',', decimal=".",names=['Date','Time','Open','High','Low','Close','Volume'],parse_dates={'datetime': ['Date', 'Time']},index_col=0)
            read_file.to_csv(new_stocks_directory+ticker+'.txt')
        elif dt.datetime.fromtimestamp(os.path.getmtime(stocks_directory+ticker+'.txt')).year==2021:
            dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H%M')
            read_file = pd.read_csv(stocks_directory+ticker+'.txt', sep=',', decimal=".",parse_dates={'datetime': ['Date', 'Time']},date_parser=dateparse,index_col=0)      
            read_file.to_csv(new_stocks_directory+ticker+'.txt')


# Loading sample of big csv:
    df = pd.read_csv("/.../US_Accidents_Dec19.csv", 
                    skiprows = lambda x: x>0 # x > 0 makes sure that the headers is not skipped 
                                        and np.random.rand() > 0.01) # returns True 99% of the time, thus skipping 99% of the time


# skip lines when reading text
    string_from_file = """
    // Author: ...
    // License: ...
    //
    // Date: ...

    Actual content...
    """

    import itertools
    for line in itertools.dropwhile(lambda line: line.startswith("//"), string_from_file.split("\n")):
        print(line)


# Merge all csv files of the same struc in the same folder (1)
    import glob
    import pandas as pd
    from time import strftime

    def folder_csv_merge(file_prefix, folder_path='', memory='no'):
        if folder_path == '':
            folder_path = input('Please enter the path where the CSV files are:\n')
        folder_path = folder_path.replace("\\","/")
        if folder_path[:-1] != "/":
            folder_path = folder_path + "/"

        file_list = glob.glob(folder_path + '*.csv')

        combined = pd.concat( [ pd.read_csv(f) for f in file_list ] )
        if memory == 'no':
            combined.to_csv(folder_path + 
                            'combined_{}_{}.csv'.format(file_prefix, 
                                                strftime("%Y%m%d-%H%M%S")), 
                            index=False)
        else:
            return combined
        print('done')


# Merge all csv files of the same struc in the same folder (2)
    import os, glob
    import pandas as pd

    stonks_directory = 'C:\\Users\\oanuf\\Data\\minute_data\\US\\Stocks_adj\\'
    combined = pd.concat([pd.read_csv(f, sep=',', decimal=".", 
                                        usecols=[0,1], 
                                        names=("Day", "Time"),
                                        nrows=1,
                                        skiprows=1,
                                        # skiprows=range(2,count-1), 
                                        header=None).
                            assign(filename = f) 
                            for f in glob.glob(stonks_directory + '*.txt')])
    combined['Symbol'] = [x.split('.')[0] for x in os.listdir(stonks_directory) if x.endswith(".txt")] # names of txt files in the directory
    combined.to_csv(stonks_directory + 'TKRS_START.csv', index=False)


# read all csvs, merge and reduce the size of big file from 30GB to 10GB   
    wdir = "C:/bigdata/pums/2014-2018/pop"
    os.chdir(wdir)
    all_files = glob.glob("*.csv")     
    pop_list = (pd.read_csv(f) for f in all_files)
    popr = pd.concat(pop_list, ignore_index=True)

    def mkdowncast(df): # reducing the size of big file from 30GB to 10GB   
        for c in enumerate(df.dtypes) : 
            if c[1] in ["int32","int64"] : 
                df[df.columns[c[0]]] = pd.to_numeric(df[df.columns[c[0]]], downcast='integer')
        for c in enumerate(df.dtypes) : 
            if c[1] in ["float64","float32"] : 
                df[df.columns[c[0]]] = pd.to_numeric(df[df.columns[c[0]]], downcast='float')
        return(df)

    poprd = mkdowncast(popr.copy())

    # Save memory of a dataframe by converting to smaller datatypes
    df = pd.read_csv("../input/titanic/train.csv", usecols = ["Pclass", "Sex", "Parch", "Cabin"])
    df.memory_usage(deep = True) # let's see how much our df occupies in memory

    # convert to smaller datatypes
    df = df.astype({"Pclass":"int8",
                    "Sex":"category", 
                    "Parch": "Sparse[int]", # most values are 0
                    "Cabin":"Sparse[str]"}) # most values are NaN
    df.memory_usage(deep = True)


# Convert quotes from csv to dictionary
    # Source: qstrader/price_handler/historic_csv_tick.py

    def _open_ticker_price_csv(self, ticker):
        ticker_path = os.path.join(self.csv_dir, "%s.csv" % ticker)
        self.tickers_data[ticker] = pd.io.parsers.read_csv(
                                                ticker_path, header=0, parse_dates=True,
                                                dayfirst=True, index_col=1,
                                                names=("Ticker", "Time", "Bid", "Ask")
                                                )


# download the zip file with many txts and move the data to 1 csv
    import requests
    url = "https://www.ssa.gov/oact/babynames/names.zip"
    
    with requests.get(url) as response:
        with open("names.zip", "wb") as temp_file:
            temp_file.write(response.content)

    data_list = [["year", "name", "gender", "count"]] # 2-dimensional Array (list of lists)

    with ZipFile("names.zip") as temp_zip: # open the zip file into memory
        for file_name in temp_zip.namelist(): # Then we read the file list.
            if ".txt" in file_name: # We will only process .txt files.
                with temp_zip.open(file_name) as temp_file: # read the current file from the zip file.
                    # The file is opened as binary, we decode it using utf-8 so it can be manipulated as a string.
                    for line in temp_file.read().decode("utf-8").splitlines():
                        line_chunks = line.split(",")
                        year = file_name[3:7]
                        name = line_chunks[0]
                        gender = line_chunks[1]
                        count = line_chunks[2]

                        data_list.append([year, name, gender, count])

    csv.writer(open("data.csv", "w", newline="", # We save the data list into a csv file.
                    encoding="utf-8")).writerows(data_list)
                    # I prefer to use writerows() instead of writerow() ...
                    # ...since it is faster as it does it in bulk instead of one row at a time.


# Download and display zipped csv data from databank.worldbank.org

    import csv, io, requests, zipfile  # noqa

    url = "http://databank.worldbank.org/data/download/WDI_csv.zip"
    filename = "WDI_Data.csv"

    # Warning: this can take two minutes to download!!
    with zipfile.ZipFile(io.BytesIO(requests.get(url).content)) as zip_file:
        print("\n".join(name for name in zip_file.namelist()))
        zip_file.extractall()

    with open(filename, newline="") as in_file:
        for row in csv.reader(in_file):
            print(", ".join(row))


# read all needed csvs from zip		

    fracfocus_url='http://fracfocusdata.org/digitaldownload/fracfocuscsv.zip'
    request = requests.get(fracfocus_url)
    zip_file = zipfile.ZipFile(io.BytesIO(request.content)) #generates a ZipFile object
    list_of_file_names = zip_file.namelist() #list of file names in the zip file
    list_to_append_to=[]
    for file_name in list_of_file_names:
        if ((file_name.endswith('.csv')) & (key_word in file_name)):
            list_to_append_to.append(file_name)
    list_of_dfs=[pd.read_csv(zip_file.open(x), low_memory=False) for x in list_to_append_to]


    def unzip(file_path, subdir):
        import zipfile
        
        #walk through folder and unzip all
        i = 0
        d = 0
        
        #if archives were inside archives
        if subdir:
            dirs = walk(file_path)
            
            for dir in dirs:
                if d > 0:
                    print "Checking "+dir[0]
                    rename_elements_in_archives(dir[0], True)
                    filenames = [f for f in listdir(dir[0]) if isfile(join(dir[0], f))]
                    for filename in filenames:
                        if "tmp" in filename:
                            try:
                                print i
                                print "Unzipping "+dir[0]+"\\"+filename
                                with zipfile.ZipFile(dir[0]+"\\"+filename, "r") as zipped:
                                    zipped.extractall(dir[0]+"\\")
                                i += 1
                            except Exception as e:
                                print e
                d += 1
        #normal way
        else:
            filenames = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        
            for filename in filenames:
                if "tmp" in filename:
                    try:
                        print "Unzipping %s" %i
                        print filename
                        with zipfile.ZipFile(file_path+filename, "r") as zipped:
                            zipped.extractall(file_path)
                            i += 1
                    except Exception as e:
                        print e


# writing multiple CSV files into one ZIP archive
    df.to_csv('collection_name.zip', compression = {'method': 'zip', 'archive_name': 'table_name.csv'}, mode='a')


# divide text (csv or ...) to small files with defined number of lines
    def splitter(name, parts = 100000):
        # make dir for files
        if not os.path.exists(name.split('.')[0]): 
            os.makedirs(name.split('.')[0])
        f = open(name, 'r', errors = 'ignore')
        lines = f.readlines()
        f.close()
        i = 0
        while i < len(lines):
            for item in lines[i:i+parts]:
                f2 = open(name.split('.')[0]+ '/'name.split('.')[0]+ str(i)+'.txt', 'a+', errors = 'ignore') 
                f2.write(item)
                f2.close()
        i += parts


# Write new csv or append existing
    df.to_csv(ticker+".csv", header=not pathlib.Path(ticker+".csv").exists(), mode='a' if pathlib.Path(ticker+".csv").exists() else 'w')


# check existing and create txts for every ticker		

    if os.path.exists('{}'.format(path)):
        response = input('A database with that path already exists. Are you sure you want to proceed? [Y/N] ')
        if response == 'Y':
            for item in os.listdir('{}/trades/'.format(path)):
                os.remove('{}/trades/{}'.format(path, item))
            os.rmdir('{}/trades/'.format(path))
            for item in os.listdir('{}'.format(path)):
                os.remove('{}/{}'.format(path, item))
            os.rmdir('{}'.format(path))
    print('Creating a new database in directory: {}/'.format(path))
    self.trades_path = '{}/trades/'.format(path)
    os.makedirs(path)
    os.makedirs(self.trades_path)
    for name in names:
        with open(self.trades_path + 'trades_{}.txt'.format(name), 'w') as trades_file:
            trades_file.write('sec,nano,name,side,shares,price\n')
					

# check latest data in file and update csv from the latest date to today

    if os.path.isfile(path):        
        df = pd.read_csv(path,index_col=0,header=0) 
        latest_date=df[df.index==max(df.index)]['DATE']
        latest= pd.datetime.strptime(latest_date[0],'%Y-%m-%d')
        ndays = pd.datetime.today().date()-latest.date()
        return str(ndays.days) + 'd'		

    DATE_FORMAT = "%Y-%m-%d"
    if os.path.isfile(fn):
        f1 = open(fn, "r")
        last_line = f1.readlines()[-1]
        f1.close()
        last = last_line.split(",")
        date = (datetime.datetime.strptime(last[0], DATE_FORMAT)).strftime(DATE_FORMAT)
        today = datetime.datetime.now().strftime(DATE_FORMAT)
        if date != today:
            with open(fn, 'a') as outFile:
                f.tail(1).to_csv(outFile, header=False)
    else:
        print("new file")
        f.to_csv(fn)

    # Get last n lines of a file:
    def tail(filename, n=10):
        with open(filename) as f:
            return collections.deque(f, n)


# Timestamped filename

    # 'sample.txt' --> 'sample_2016_01_23_18_07_23.txt'
    # YYYY_MM_DD_hh_mm_ss order sorts oldest to newest.

    import datetime
    import os

    def timestamped_filename(file_name, date_time=None):
        date_time = date_time or datetime.datetime.now()
        root, ext = os.path.splitext(file_name)
        fmt = "{}{:_%Y_%m_%d_%H_%M_%S}{}"
        return fmt.format(root, date_time, ext)

    if __name__ == "__main__":
        import time

        print(timestamped_filename("sample.txt"))
        time.sleep(1)
        print(timestamped_filename("sample.txt"))


# Keep track of where your data is coming when you are using multiple sources

    # let's generate some fake data
    df1 = generate_sample_data()
    df2 = generate_sample_data()
    df3 = generate_sample_data()
    df1.to_csv("trick78data1.csv")
    df2.to_csv("trick78data2.csv")
    df3.to_csv("trick78data3.csv")

    # Step 1 generate list with the file name
    lf = []
    for _,_, files in os.walk("/kaggle/working/"):
        for f in files:
            if "trick78" in f:
                lf.append(f)
                
    lf

    # You can use this on your local machine
    #from glob import glob
    #files = glob("trick78.csv")

    # Step 2: assing create a new column named filename and the value is file
    # Other than this we are just concatinating the different dataframes
    df = pd.concat((pd.read_csv(file).assign(filename = file) for file in lf), ignore_index = True)
    df.sample(10)


# Scrape tables from many pages and put every result to separate csv 

    for ticker in ticker_list:
        
        output_filename = "dividends\{0}.csv".format(ticker)     # set the output file to be `$ticker.csv` in the folder `dividends`
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)     # Check if the folder exists, if not make it
        output_file = open(output_filename, 'w')     # make/open the ticker file

        url = "https://www.tickertech.net/bnkinvest/cgi/?n=2&ticker=" + ticker + "&js=on&a=historical&w=dividends2"
        req = requests.get(url)
        soup = BeautifulSoup(req.content, "lxml")
        # find the one table we are looking for with dividend data, and get all table rows from that table
        dividend_rows = soup.find("table", attrs={"bgcolor": "#EEEEEE"}).find_all("tr")
        for row in dividend_rows:
            columns = list(row.stripped_strings) # extract all the strings from the row
            columns = [x for x in columns if 'allow' not in x] # remove the lingering javascript rows
            if len(columns) == 2: # if there are only 2 columns (date, ratio), ...
                output_file.write("{0}, {1} \n".format(columns[0], columns[1])) # ... the data is correct and we can write it
        output_file.close()


# Reading big file with modin
    # pip install modin[ray]
    
    Alpaca_directory = 'D:\\Data\\minute_data\\US\\alpaca_ET_adj\\gesamt\\'
    s = time.time()
    alpaca_quotes = pd.read_csv(Alpaca_directory+"Alpaca_min_quotes_ET_adj.csv",index_col='timestamp', parse_dates=['timestamp'])
    e = time.time()
    print("Pandas Loading Time = {}".format(e-s)) # 190 secs

    import modin.pandas as pd
    s = time.time()
    alpaca_quotes = pd.read_csv(Alpaca_directory+"Alpaca_min_quotes_ET_adj.csv",index_col='timestamp', parse_dates=['timestamp'])
    e = time.time()
    print("Modin Loading Time = {}".format(e-s)) # 107 secs


# Read big files with pandas 2.0
    df = pd.read_csv("data/hn.csv") # 12s
    df_arrow = pd.read_csv("data/hn.csv", engine='pyarrow', dtype_backend='pyarrow') # 329ms
    df.info() # Investigating the dtypes of each DataFrame

# read big files with parquet
    df = pd.read_csv("large.csv") # 13 sec for 700mb file
    df = pd.read_csv("large.csv", engine="pyarrow") # available in from pd 1.4; 6 sec
    df = pd.read_csv("large.csv")
    df.to_parquet("large.parquet", compression=None)
    df = pd.read_parquet("large.parquet", engine="fastparquet") # 2 sec


# Read - write to Azure blob

    # Read csv/xlsx from blob
        company_data="ITR_Tool_Sample_Data_Small.xlsx" # this file is provided initially
        company_data_URL = f'https://{STORAGE_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{company_data}?{BLB_SAS}' # azure
        tab_company_data_and_emissions = "ITR input data"
        data = pd.read_excel(company_data_URL,sheet_name=tab_company_data_and_emissions)

    # Read json from blob
        benchmark_prod_json_file = "benchmark_production_OECM.json"
        benchmark_prod_json = f'https://{STORAGE_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{benchmark_prod_json_file}?{BLB_SAS}' # azure
        with urllib.request.urlopen(benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)

    # write panda to csv to blob  

        # pip install azure-storage-blob
        from azure.storage.blob import ContainerClient, BlobServiceClient

        def write_csv(df_path, df):
            container_client = ContainerClient(env['container_url'],container_name=CONTAINER_NAME,credential=env['container_cred'])
            output = df.to_csv(index_label="idx", encoding = "utf-8")
            blob_client = container_client.get_blob_client(df_path)
            blob_client.upload_blob(output, overwrite=True)
            return 'success'

        PRICEDOMSIZE=  5  # domain size of prices
        SIZEDOMSIZE= 100
        def createTable(N):
            return pd.DataFrame({
                    'pA': np.random.randint(0, PRICEDOMSIZE, N),
                    'pB': np.random.randint(0, PRICEDOMSIZE, N),
                    'sA': np.random.randint(0, SIZEDOMSIZE, N),
                    'sB': np.random.randint(0, SIZEDOMSIZE, N)})
        temp_df = createTable(5)

        def PandaToBlobStorage(dataframe=None, filename=None):
            upload_file_path = os.path.join('OUTPUT', f"{filename}.csv")
            blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=upload_file_path)
            output = dataframe.to_csv(index=False, encoding="utf-8")
            blob_client.upload_blob(output, blob_type="BlockBlob")
            return 'success'

        PandaToBlobStorage(temp_df,'test_df')

    # Write local file to Blob

        def LocalFileToBlobStorage(file_path,file_name):
            blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=file_name)
            with open(file_path,'rb') as data:
                blob_client.upload_blob(data)
            print(f'Uploaded {file_name}')
            
        LocalFileToBlobStorage('F:\\ITR_Dash\\requirements.txt','requirements.txt')
