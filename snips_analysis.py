import pandas as pd
import numpy as np
import datetime

# Series creation
		pd.Series([1,2,3,4])
		pd.Series([3,4,5,6,7],index=['a','b','c','d','e'])
		pd.Series([3]*5)
		np.zeros(10)
		pd.Series(np.arange(4,9)) # using the numpy function
		pd.Series(np.linspace(0,9,5)) # allows to specify the number of values to be created btw boundaries

		pd.Series(np.random.normal(size=5))
		np.random.randint(50,101,10)

        result["Asset"].unique().tolist()

		a = np.array([4] * 16)
		a[1::] = [42] * 15
		a[1:8:2] = 16


		import pandas.util.testing as tm
		tm.N, tm.K = 5,3
		tm.makeFloatSeries(), tm.makeBoolIndex(), tm.makeCategoricalIndex()
		tm.makeCustomIndex(nentries=4,nlevels=2), tm.makeFloatIndex(), tm.makeIntIndex()
		tm.makeMultiIndex(), tm.makeRangeIndex(), tm.makeIntervalIndex()

		# All possible combinations (Permutations)
			from itertools import permutations 
			my_list = [1,2,3]
			perm = list(permutations(my_list))

				#(1, 2, 3)
				#(1, 3, 2)
				#(2, 1, 3)
				#(2, 3, 1)
				#(3, 1, 2)
				#(3, 2, 1)


# Create dataframes 
		dates=pd.date_range('2016-08-01','2017-08-01')
		s = pd.Series(np.random.randint(50,60,size=len(dates)))
		temp_df=pd.DataFrame({'Hurra':dates, 'Pinguin':s})
		

		N, M = 20, 4
		A = np.random.randn(N, M) + np.arange(M)/4 # generate a random array, add a small constant to each column
		df = pd.DataFrame(A) # converts numpy array to pandas df


        seq = np.random.choice(['make', 'miss'], size=10, p=[.65, .35])
        streaks = pd.Series(seq, name='result').to_frame()


		# Generate and plot random walks
			# Generate normally distributed errors
			randos = [np.random.randn(100) for i in range(100)]
			y = np.random.randn(100)
			# Generate random walks
			randows = [[sum(rando[:i+1]) for i in range(100)] for rando in randos]
			yw = [sum(y[:i+1]) for i in range(100)]

			plt.figure(figsize=(15,7))
			for i in range(100):
				plt.plot(randows[i], alpha=0.5)
			plt.show()  


		# create a dataframe
		PRICEDOMSIZE=  5  # domain size of prices
		SIZEDOMSIZE= 100
		def createTable(N):
			return pd.DataFrame({
					'pA': np.random.randint(0, PRICEDOMSIZE, N),
					'pB': np.random.randint(0, PRICEDOMSIZE, N),
					'sA': np.random.randint(0, SIZEDOMSIZE, N),
					'sB': np.random.randint(0, SIZEDOMSIZE, N)})
		createTable(5)

		# quickly create a dataframe for testing
		import pandas.util.testing as tm
		tm.N, tm.K = 5,3
		tm.makeDataFrame(), tm.makeMixedDataFrame(), tm.makeTimeDataFrame(freq="W")


		lst = [40, 10, 20, 30]
		names = ['AAA','Adfsdf','dfwef','fwefw']
		temp_df=pd.DataFrame(lst)
		temp_df=pd.DataFrame(list(zip(names,lst)),columns=["Name","Age"])
		
		# Create blank dataframe: could be useful if we want to append data row by row to a Dataframe.
		# In that case it’s better to have predefined columns
		blank_df=pd.DataFrame(columns=["Name","Age"])


		# Create rows for values separated by commas in a cell
			d = {"Team":["FC Barcelona", "FC Real Madrid"], 
				"Players":["Ter Stegen, Semedo, Piqué, Lenglet, Alba, Rakitic, De Jong, Sergi Roberto, Messi, Suárez, Griezmann",
						"Courtois, Carvajal, Varane, Sergio Ramos, Mendy, Kroos, Valverde, Casemiro, Isco, Benzema, Bale"]}
			df = pd.DataFrame(d)
			df.assign(Players = df["Players"].str.split(",")).explode("Players")


# Importing data 
		SPX500=pd.read_csv("D:\\Data\\tick_data\\tick_data_zorro\\SPX500_2015.csv")
		SPY_TICK=pd.read_csv("D:\\Data\\tick_data\\SPY_TICK_TRADE.csv")
		GBPCAD=pd.read_csv("D:\\Data\\minute_data\\GBPCAD_2017.csv", header=False, parse_dates=['Date'])
		NQ100=pd.read_csv("http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx?render=download",
						usecols=[0,1,2,5],
						index_col='Symbol',
						skipinitialspace=True)

		dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')
		AAPL = pd.read_csv("D:\\Data\\minute_data\\AAPL.txt", 
                            sep='\t', decimal=",", 
                            parse_dates={'datetime': ['Date', 'Time']}, date_parser=dateparse)


		pd.read_csv('http://www.nasdaq.com/investing/etfs/etf-finder-results.aspx?download=Yes')['Symbol'].values
		calls_df, = pd.read_html("http://apps.sandiego.gov/sdfiredispatch/", header=0, parse_dates=["Call Date"])

		df = pd.read_csv("name_age.csv", na_values=["na", "not available"])

		def CompleteGender(cell):
    			if cell=='m':
    					return 'Male'
				elif cell = 'f':
    					return 'Female'
				else:
    					return 'NA'
		df_new = pd.read_csv("name_age.csv", converters = {"gender" : completeGender})


        # read parqueat is much faster than csv
            df = pd.read_csv("large.csv") # 13 sec for 700mb file
            df = pd.read_csv("large.csv", engine="pyarrow") # available in from pd 1.4; 6 sec
            df = pd.read_csv("large.csv")
            df.to_parquet("large.parquet", compression=None)
            df = pd.read_parquet("large.parquet", engine="fastparquet") # 2 sec



		# from quandl
			import quandl
			start = datetime.datetime(2016,1,1)
			end = datetime.date.today()
			
			apple, microsoft, google = (quandl.get("WIKI/" + s, start_date=start, end_date=end) 
										for s in ["AAPL", "MSFT", "GOOG"])
			
			stocks = pd.DataFrame({"AAPL": apple["Adj. Close"],
								"MSFT": microsoft["Adj. Close"],
								"GOOG": google["Adj. Close"]})

			spyderdat = pd.read_csv("/home/curtis/Downloads/HistoricalQuotes.csv") # from http://www.nasdaq.com/symbol/spy/historical
			spyderdat = pd.DataFrame(spyderdat.loc[:, ["open", "high", "low", "close", "close"]]
									.iloc[1:].as_matrix(),
									index=pd.DatetimeIndex(spyderdat.iloc[1:, 0]),
									columns=["Open", "High", "Low", "Close", "Adj Close"])
						.sort_index()
			spyder = spyderdat.loc[start:end]
			stocks = stocks.join(spyder.loc[:, "Adj Close"])
							.rename(columns={"Adj Close": "SPY"})


# Analysing series -------------------------------------------------------

		bob = pd.Series(np.arange(3,30,3))
		(bob>10).any() # bool
		(bob>2).all() # bool
		(bob>15).sum()
		bob.isnull().sum()

		np.nonzero([1,2,0,0,4,0]) # find indices of non-zero elements


		def count_occurrences(lst, val):
		      return len([x for x in lst if x == val and type(x) == type(val)])
		count_occurrences([1, 1, 2, 1, 2, 3], 1) # 3


		li = [1,2,3,'a','b','c']
		'a' in li # true

		s = input()
		up    = len([i for i in ['1', '2', '3'] if i in s]) ==  0 # true if s does not contains one of these
		right = len([i for i in ['3', '6', '9', '0'] if i in s]) == 0
		left  = len([i for i in ['1', '4', '7', '0'] if i in s]) == 0
		down  = len([i for i in ['7', '9', '0'] if i in s]) == 0
		ok = not(up or right or left or down) #if the is at least one true then it is not uniqe and print NO
												#if all of them are 0 then not 0 == true and print YES
		if ok:
			print("YES")
		else: #not ok
			print("NO")

		arr = [1, 2, 13, 10, 320, 34, 0, 10, 0, 4, 5, 16, 0, 0, 0, 0, 9, 8, 7, 0, 10, 11]
		np.diff(arr)>10
		np.equal(arr, 10).view(np.int8)


		# the most often occuring names using collection.Counter
			from collections import Counter
			cheese = ["gouda", "brie", "feta", "cream cheese", "feta", "cheddar",
					"parmesan", "parmesan", "cheddar", "mozzarella", "cheddar", "gouda",
					"parmesan", "camembert", "emmental", "camembert", "parmesan"]
			cheese_count = Counter(cheese) # maps items to number of occurrences
			# use update(more_words) method to easily add more elements to counter
			print(cheese_count.most_common(3)) # Prints: [('parmesan', 4), ('cheddar', 3), ('gouda', 2)]

		# Takes in an array of numbers and finds consecutive runs of the number to_find
			def find_runs(arr, to_find):
				# Create an array that is 1 where arr is equal to to_find, and pad each end with an extra 0.
				is_the_number = np.concatenate(([0], np.equal(arr, to_find).view(np.int8), [0]))
				absdiff = np.abs(np.diff(is_the_number))
				ranges = np.where(absdiff == 1)[0].reshape(-1, 2) # Runs start and end where absdiff is 1.
				return ranges
			arr = [1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 9, 8, 7, 0, 10, 11]
			to_find=3
			print("find_runs() output: ")
			print(find_runs(test_arr, 0))
			print("Array outputted should be equal to: ")
			print("[[3, 9], [12, 16], [19, 20]]")


		# extract a list of the companies paying ...
		# ...below minimum wage (< $9) for at least one employee
			companies = {
					'CoolCompany' : {'Alice' : 33, 'Bob' : 28, 'Frank' : 29},
					'CheapCompany' : {'Ann' : 4, 'Lee' : 9, 'Chrisi' : 7},
					'SosoCompany' : {'Esther' : 38, 'Cole' : 8, 'Paris' : 18}}
			illegal = [x for x in companies if any(y<9 for y in companies[x].values())]


		# find cities with above-average pollution peaks
			X = np.array(
						[[ 42, 40, 41, 43, 44, 43 ], # Hong Kong
						[ 30, 31, 29, 29, 29, 30 ], # New York
						[ 8, 13, 31, 11, 11, 9 ], # Berlin
						[ 11, 11, 12, 13, 11, 12 ]]) # Montreal
			cities = np.array(["Hong Kong", "New York", "Berlin", "Montreal"])
			polluted = set(cities[np.nonzero(X > np.average(X))[0]])
					# "X > np.average(X)" - element-wise comparison
					# "np.nonzero(...)" # True = 1,
					# [0] - the above produces two tuples, 
						# the first giving the row indices of nonzero elements,
						# and the second giving their respective column indices
					# set(..) # remove duplicates


		# find the closest value (to a given scalar)
			Z = np.arange(100)
			v = np.random.uniform(0,100)
			index = (np.abs(Z-v)).argmin()
			print(Z[index])


# Comparing several series / dataframes -------------------------------------------------------

		# find common values between two arrays
            Z1 = np.random.randint(0,10,10)
            Z2 = np.random.randint(0,10,10)
            print(np.intersect1d(Z1,Z2))

        # difference btw 2 dictionaries
            dict1 = {1:'donkey', 2:'chicken', 3:'dog'}
            dict2 = {1:'donkey', 2:'chimpansee', 4:'chicken'}
            set1 = set(dict1.items())
            set2 = set(dict2.items())
            set1 ^ set2

		# find common values between two arrays
            dctA = {'a': 1, 'b': 2, 'c': 3}
            dctB = {'b': 4, 'c': 3, 'd': 6}		
            for ky in dctA.keys() & dctB.keys(): # loop over dicts that share (some) keys
                print(ky)
            for item in dctA.items() & dctB.items(): # loop over dicts that share (some) keys and values
                print(item)

		# Check if arrays are equal
            A= np.random.randint(0,2,5)
            B= np.random.randint(0,2,5)
            equal = np.allclose(A,B) # Assuming identical shape of the arrays and a tolerance for the comparison of values 
            equal = np.array_equal(A,B) # Checking both the shape and the element values, no tolerance (values have to be exactly equal)



		def difference(a, b):
			_b = set(b)
			return [item for item in a if item not in _b]
		difference([1, 2, 3], [1, 2, 4]) # [3]

		def all_equal(lst):
    			return lst[1:] == lst[:-1]
		all_equal([1, 2, 3, 4, 5, 6]) # False
		all_equal([1, 1, 1, 1]) # True

		def all_unique(lst):
			return len(lst) == len(set(lst))
		x = [1, 2, 3, 4, 5, 6]
		y = [1, 2, 2, 3, 4, 5]
		all_unique(x) # True
		all_unique(y) # False


	# Comparison of dataframes
		def overlap_by_symbol(old_df: pd.DataFrame, new_df: pd.DataFrame, overlap: int):
			"""
			Overlap dataframes for timestamp continuity. 
			Prepend the end of old_df to the beginning of new_df, grouped by symbol.
			If no symbol exists, just overlap the dataframes
			:param old_df: old dataframe
			:param new_df: new dataframe
			:param overlap: number of time steps to overlap
			:return DataFrame with changes
			"""
			if isinstance(old_df.index, pd.MultiIndex) and isinstance(new_df.index, pd.MultiIndex):
				old_df_tail = old_df.groupby(level='symbol').tail(overlap)

				old_df_tail = old_df_tail.drop(set(old_df_tail.index.get_level_values('symbol')) - set(new_df.index.get_level_values('symbol')), level='symbol')

				return pd.concat([old_df_tail, new_df], sort=True)
			else:
				return pd.concat([old_df.tail(overlap), new_df], sort=True)

	# Compare 2 datasets of quotes: inner merger
		def pair_data_verifier(array_df_data, pair_tickers, threshold=10):
			"""
			merge two dataframes, 
			verify if we still have the same number of data we originally had.
			use an inputted threshold that tells us 
			whether we've lost too much data in our merge or not.
			threshold: max number of days of data 
					we can be missing after merging
			"""
			stock_1 = pair_tickers[0]
			stock_2 = pair_tickers[1]
			df_merged = pd.merge(array_df_data[0], array_df_data[1], 
								left_on=['Date'], right_on=['Date'], 
								how='inner')
			
			new_col_names = ['Date', stock_1, stock_2] 
			df_merged.columns = new_col_names
			# round columns
			df_merged[stock_1] = df_merged[stock_1].round(decimals = 2)
			df_merged[stock_2] = df_merged[stock_2].round(decimals = 2)
			
			new_size = len(df_merged.index)
			old_size_1 = len(array_df_data[0].index)
			old_size_2 = len(array_df_data[1].index)

			print("Pairs: {0} and {1}".format(stock_1, stock_2))
			print("New merged df size: {0}".format(new_size))
			print("{0} old size: {1}".format(stock_1, old_size_1))
			print("{0} old size: {1}".format(stock_2, old_size_2))

			if (old_size_1 - new_size) > threshold or (old_size_2 - new_size) > threshold:
				print("This pair {0} and {1} were missing data.".format(stock_1, stock_2))
				return False
			else:
				return df_merged


# Changing series -------------------------------------------------------

		AAPL.set_index('datetime',inplace=True) # "inplace" make the changes in the existing df

		goa=Series(np.random.normal(size=5))
		goa[5]=100 # changing
		del(goa[2]) # deleting

		Z = np.random.random(10)
		Z[Z.argmax()] = 0 # replace the maximum value by 0


		d1 = {'a': 1}
		d2 = {'b': 2}
		d1.update(d2)
		print(d1)

		# remove duplicates
			li = [3, 2, 2, 1, 1, 1]
			list(set(li)) #=> [1, 2, 3]

        # Filter even values out
            li = [1,2,3,4,5,6,7,8,9,10]
            [i for i in li if i % 2 != 0]
            #=> [1, 3, 5, 7, 9]


        # remove negatives
            def remove_negatives(x):
                return True if x >= 0 else False
                
            a = [-10, 27, 1000, -1, 0, -30][x for x in filter(remove_negatives, a)] 
            #=> [27, 1000, 0]

		# replace the even with previous
			visitors = ['Firefox', 'corrupted', 'Chrome', 'corrupted',
						'Safari', 'corrupted', 'Safari', 'corrupted',
						'Chrome', 'corrupted', 'Firefox', 'corrupted']
			visitors[1::2] = visitors[::2] 


		# if you suggest 20%: neglect the best 10% of values and the worst 10% of values
			def trimmean(arr, per):
				ratio = per/200 # /100 for easy calculation by *, and /2 for easy adaption to best and worst parts.
				cal_sum = 0 # sum value to be calculated to trimmean.
				arr.sort()
				neg_val = int(len(arr)*ratio)
				arr = arr[neg_val:len(arr)-neg_val]
				for i in arr:
					cal_sum += i
				return cal_sum/len(arr)


# describe DF ---------------------------------------------------------------------

		type(NQ100['lastsale'])
		SPY_TICK.dtypes
		AAPL.dtypes
		SPY_TICK.shape
		rows, columnd = AAPL.shape
		len(AAPL)
		AAPL.columns

		AAPL.count()
		SPY_TICK.describe()
		SPY_TICK.info()

		SPY_TICK.describe()
        
        df.apply(lambda x: x.nunique()) # number of distinct values in each column

        all_hist_capital['symbol'].value_counts() # counts how many times each symbol appears

        data.hist(figsize=(14,14), xrot=45)
        plt.show()
        df['Age'].hist(figsize = (15,5))


        for column in data.select_dtypes(include='object'):
            if data[column].nunique() < 10:
                sns.countplot(y=column, data=data)
                plt.show()

		df.groupby("continent")["beer_servings"].describe()


		AAPL['Volume'].mean()
		AAPL['Volume'].std() # mix(), max(), std()



		AAPL.isna().mean() # calculate the % of missing values in each row
		SPY_TICK.isna().mean() # calculate the % of missing values in each row



        read_file[read_file.isna().any(axis=1)].count()
        read_file[read_file.isnull().any(axis=1)].count()

        # find duplicates 
        Index_const[Index_const.duplicated(subset=['symbol'],keep=False)]


		# select columns:
		list(my_dataframe)
		AAPL.columns.values.tolist()


        # Calculating streaks

            seq = np.random.choice(['make', 'miss'], size=10, p=[.65, .35])
            streaks = pd.Series(seq, name='result').to_frame()
            streaks['start_of_streak'] = streaks.result.ne(streaks['result'].shift()) # identify the start of each streak
                                                # Series.ne() compares original and shifted series and tell which are not equal.
            streaks['streak_id'] = streaks['start_of_streak'].cumsum() # treat True as 1 and False as 0
            streaks['streak_counter'] = streaks.groupby('streak_id').cumcount() + 1




		# describe DF
		pip install pandas-profiling 
		import pandas_profiling
		AAPL.profile_report() # Show in NB
		profile = AAPL.profile_report(title='AAPL_data_report')  
		profile.to_file(outputfile="AAPL_data_report.html")


		# most often occuring names using collection.Counter
			from collections import Counter
			cheese = ["gouda", "brie", "feta", "cream cheese", "feta", "cheddar",
					"parmesan", "parmesan", "cheddar", "mozzarella", "cheddar", "gouda",
					"parmesan", "camembert", "emmental", "camembert", "parmesan"]
			cheese_count = Counter(cheese) # Counter is just a dictionary that maps items to number of occurrences
			# use update(more_words) method to easily add more elements to counter
			print(cheese_count.most_common(3)) # Prints: [('parmesan', 4), ('cheddar', 3), ('gouda', 2)]


		# memory usage
			AAPL.info(memory_usage = "deep") # Show the global usage of memory of the df"
			AAPL.memory_usage(deep = True) # Show the usage of memory of every column


# changes to DF ---------------------------------------------------------------------

		# Rename columns
			NQ100.rename(columns={'lastsale':'Last'})
            constituents.rename(columns={'Constituent RIC': 'ric', 'Constituent Name': 'name'}, inplace=True)

            SP500_indicators.columns.values[11] = "3MAvgVolume"
            SP500_indicators.columns.values[-11] = "EarnScore_Date"


			# add a prefix or suffix to all columns
			df.add_prefix("1_")
			df.add_suffix("_Z")

			col_names = {"Positionsinhaber": "Holder",
						"Emittent": "Issuer",
						"Datum": "Date"}
			df.rename(columns = col_names, inplace = True)

            return_var=return_var.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'})
            return_var = return_var[['Open','High','Low','Close','Volume']] # Change order of columns

            df.columns = map(str.lower, list(df.columns)) # convert headers to lower type


        # Dealing with N/A-s
        
            # different fillna for every column
                df.fillna({	'temp':0,
                            'wind':0,
                            'status':'sunny'})
                df.fillna(df.mean())
                df.fillna(df.mean()['temperature':'windSpeed'])

            # Detect and Replace Missing Values
                if df.isnull().values.any():
                    affected_cols = [col for col in df.columns if df[col].isnull().any()]
                    affected_rows = df.isnull().sum()
                    missing_list = []
                    for each_col in affected_cols:
                    missing_list.append(each_col)
                for each in missing_list:
                    df[each] = df[each].interpolate()


		# Deleting

			# delete columns
				
                df.drop('column_name', axis=1, inplace=True)

                UC.drop(UC.columns[[3,4]],axis=1)
				interesting_collums = ['loyalty', 'satisfaction','educ']      
				reduced = data[interesting_collums]

				AAPL=AAPL.drop(AAPL.columns[[-1]],axis=1) # delete last column
                return_var=return_var.drop(return_var.columns[[5,6]],axis=1) # delete last 2 columns


				import itertools
				datecols = ['year', 'month', 'day']
				df = pd.DataFrame(list(itertools.product([2016,2017],[1,2],[1,2,3])),columns = datecols)
				df['data']=np.random.randn(len(df))
				df.index = pd.to_datetime(df[datecols])
				df=df.drop(datecols,axis=1).squeeze()
				df.dropna(axis = "columns") # drop any column that has missing values
				df.dropna(thresh = len(df)*0.95, axis = "columns") # drop column where missing values are above a threshold

				# Remove a column and store it as a separate series
				meta = df.pop("Metascore").to_frame() 

                # Delete empty columns
                for column in df.columns:
        			if pd.isnull(df[column]).all():
		        		df=df.drop(column,1)

			# delete rows
				NQ100_small.drop('PYPL')
				NQ100new[-NQ100new.Last>1000]
				UC=USDCHF.dropna()
				df1.dropna(axis = "rows") # drop any row that has missing values

                # delete rows where the date belong to rare dates
                unique_dates= all_hist_capital['date'].value_counts()
                all_hist_capital = all_hist_capital[~all_hist_capital['date'].isin(unique_dates[unique_dates < 5].index)] 

                # delete outliers
                no_outlier_prices = prices_only[(np.abs(stats.zscore(prices_only)) < 6).all(axis=1)]


		# Appending & Changing rows
			UC_new= UC.nlargest(20,'Volume')
			UC_new.append(UC.nlargest(20,'Minute_ClCl'))

			UC_new.loc['ZABR']=['Ukraine',100,120,1,2,3] # the number of columns should match

			ho = ['ON']  # overnight
			ho.extend([str(i) + 'W' for i in range(1,4)])  # weekly tenors
			ho.extend([str(i) + 'M' for i in range(1,12)])  # monthly tenors
			ho.extend([str(i) + 'Y' for i in range(1,51)])  # yearly tenors


            # Anonymize data (change real names to fake)
                def anonymize(portfolio, provider):
                    portfolio_companies = portfolio['company_name'].unique()
                    for index, company_name in enumerate(portfolio_companies):
                        portfolio.loc[portfolio['company_name'] == company_name, 'company_id'] = 'C' + str(index + 1)
                        portfolio.loc[portfolio['company_name'] == company_name, 'company_isin'] = 'C' + str(index + 1)
                        provider.data['fundamental_data'].loc[provider.data['fundamental_data']['company_name'] == company_name, 'company_id'] = 'C' + str(index + 1)
                        provider.data['fundamental_data'].loc[provider.data['fundamental_data']['company_name'] == company_name, 'company_isic'] = 'C' + str(index + 1)
                        provider.data['target_data'].loc[provider.data['target_data']['company_name'] == company_name, 'company_id'] = 'C' + str(index + 1)
                        portfolio.loc[portfolio['company_name'] == company_name, 'company_name'] = 'Company' + str(
                            index + 1)
                        provider.data['fundamental_data'].loc[provider.data['fundamental_data']['company_name'] == company_name, 'company_name'] = 'Company' + str(
                            index + 1)
                        provider.data['target_data'].loc[provider.data['target_data']['company_name'] == company_name, 'company_name'] = 'Company' + str(
                            index + 1)
                    for index, company_name in enumerate(provider.data['fundamental_data']['company_name'].unique()):
                        if company_name not in portfolio['company_name'].unique():
                            provider.data['fundamental_data'].loc[provider.data['fundamental_data']['company_name'] == company_name, 'company_id'] = '_' + str(index + 1)
                            provider.data['fundamental_data'].loc[provider.data['fundamental_data']['company_name'] == company_name, 'company_name'] = 'Company_' + str(
                                index + 1)
                    return portfolio, provider


            # Change format of columns
                result['EntryTime'] = pd.to_datetime(result['EntryTime'], format= '%Y/%m/%d %H:%M')

                impactful_data['timestamp_af'] = impactful_data['timestamp'].apply(lambda x: self.utc_to_local(x))

                cols_to_round = SP500_indicators.select_dtypes(include=['float64']).columns.to_list() # selecting all columns of type float
                SP500_indicators[cols_to_round] = SP500_indicators[cols_to_round].astype('Int64') # tranform selected columns to integer (I don't really care about decimals)

    			data['educ'] = pd.to_numeric(data['educ'],errors='coerce') # errors='coerce' means that we force the conversation. noncovertable are set to NaN


			# Adding anomalies to df
                anomaly_dictionary={80: 3.1, 200: 3, 333: 1, 600: 2.6, 710: 2.1}
                gasoline_price_df.loc[:,'Artificially_Generated_Anomaly']=0
                for index, anomaly_value in anomaly_dictionary.items():
                    gasoline_price_df.loc[index,'Gasoline_Price']=anomaly_value
                    gasoline_price_df.loc[index,'Artificially_Generated_Anomaly']=1


			# delete $ from string
				df.state_bottle_retail.str.replace('$','') # 4.5*X ms: replaces the ‘$’ with a blank space for each item in the column
				df.state_bottle_retail.apply(lambda x: x.replace('$','')) # 4*X ms: pandas ‘apply’ method, which is optimized to perform operations over a pandas column
				df.state_bottle_retail.apply(lambda x: x.strip('$')) # 3*X ms: strip does one less operation: just takes out the ‘$.’
				df.state_bottle_retail = [x.strip('$') for x in df.state_bottle_retail] # 2*X ms: list comprehension
				df.state_bottle_retail = [x[1:] for x in df.state_bottle_retail] # X ms: built in [] slicing, [1:] slices each string from 2nd value till end


            # remove part of string from several columns
                columns_string =['EarnScore_Date','Expect_Rep_Date','MomScore_Date','AvgScore_Date'] # select columns with dates (but actually the dates are strings)
                SP500_indicators[columns_string] = SP500_indicators[columns_string].replace('T00:00:00Z', '', regex=True) # removing not needed part from strings


			# Remove dubpicates from rows:
			   # 	0  1    2    3
			   # 0  A  B    C    D
			   # 1  A  D    C  NaN
			   # 2  C  B  NaN  NaN
			   # 3  B  A  NaN  NaN

				pd.DataFrame(list(map(pd.unique, df.values)))
				pd.DataFrame(df.apply(pd.Series.unique, axis=1).tolist())


		# creating new columns
			NQ100['Capitalisation']=NQ100.Last*NQ100.share_volume
			USDCHF['Minute_ClCl']=USDCHF.Close.diff()
			AAPL['Week_Vol']=AAPL["Volume"].rolling(5)
			apple["20d"] = np.round(apple["Adj. Close"].rolling(window = 20, center = False).mean(), 2)

            qvdf['comboaccrual']=qvdf[["p_snoa","p_sta"]].mean(axis=1)

            df_ranked['total_rank']= df_ranked.iloc[:, -4:-1].mean(axis=1)


            # create col in specific place
                SP500_indicators.insert(loc=11, column='Trades_rank', value=SP500_indicators['Trades'].rank(ascending=False)) # rank and create new column in specific location (after 'Trades')

            # Comparing previous row values
                df['match'] = df.col1 == df.col1.shift()
                df['match'] = df.col1.eq(df.col1.shift())
                df['match'] = df['col1'].diff().eq(0)
                def comp_prev(a):
                    return np.concatenate(([False],a[1:] == a[:-1]))
                df['match'] = comp_prev(df.col1.values)


            # new column with value depending on previous row
            trading_days_df['DOM'] = np.where(trading_days_df.date.dt.to_period('M') != trading_days_df.date.shift().dt.to_period('M'), 'FDM', 
                                    np.where(trading_days_df.date.dt.to_period('M') != trading_days_df.date.shift(-1).dt.to_period('M'), 'EOM', 'No'))   # FDM - 1st day of month, EOM - end of Month



            # create new col "rank": rank of market cap among symbols on every date
            qvdf["rank"] = qvdf.groupby("date")["marketCap"].rank(ascending=False)

			NQ100['Random']=Series(np.random.normal(size=len(NQ100)),index=NQ100.index)
			NQ100.insert(1,'Rand',Series(np.random.normal(size=len(NQ100)),index=NQ100.index))
			NQ100.Randomize=NQ100.Rand

            # Categorise
                result['GapSize_Rank']=pd.qcut(abs(result["GapSize"]), q = 3, labels = ["small", "avg", "big"])
                df['color'] = df.Set.map( lambda x: 'red' if x == 'Z' else 'green')
                df['color'] = np.where(df['Set']=='Z', 'green', 'red')
                result['Strategy'] = np.where(result['EntryTime'].dt.dayofweek == 6, 'Mom', 'Revers')
                apple["Regime"] = np.where(apple['20d-50d'] > 0, 1, 0) # np.where() is a vectorized if-else function

                impactful_data['major'] = impactful_data['economy'].apply(lambda x: True if x in major else False)

			
            hist_data['curr_pair'] = hist_data[0].apply(lambda x: x[10:16])

            read_file['Date'] = pd.to_datetime(read_file['Date']+' '+read_file['Time'] # merge 2 columns
                                    , format='%d.%m.%Y %H:%M') # adding format makes converting much faster

            gasoline_price_df.loc[:,'Artificially_Generated_Anomaly']=0

            ETF_minute['Start_New'] = ETF_minute['Start_New'].str.split("'").str[1] # there is "b" in the beginning, which is byte type, 


			# split column into 2 columns
    			df[[one,two]] = df[orig].str.split(separator,expand=True)


        # restructure
            df_wide = pd.DataFrame({"Country": ["France", "US", "UK"],
                                    "22/01/2020": [1,2,3],
                                    "23/01/2020": [4,5,6],
                                    "24/01/2020": [7,8,9],
                                    "25/01/2020": [10,11,12],
                                    "26/01/2020": [13,14,15],
                                    }
                                    )
            df_wide
            df_wide.melt(id_vars='Country')
            df_wide.melt(id_vars='Country',var_name='Date',value_name='Cases')
            

# Extracting sub-set from DF --------------------------------------------------------------

        # Exctract from series
            s = pd.Series(np.random.randint(10,100,size =30))
            s[3], s[[1,3]], s[3:17:4] # step 4
            s.head(), s[:5], s[:-10] # all but the last 10

            bob = pd.Series(np.arange(3,30,3))
            bob >15
            bob[(bob>15) & (bob<25)]

            a = np.array([2,4,6,9,4])
            np.argmax(a)
            np.argwhere(a==4)

            line = pd.Series(np.random.randint(1,200,size=1000))
            line.sample(n=3)
            line.sample(frac=0.01) #selects 1% of data


            # Split based on file extension
                from more_itertools import partition
                files = ["foo.jpg", "bar.exe", "baz.gif", "text.txt", "data.bin"]
                ALLOWED_EXTENSIONS = ('jpg','jpeg','gif','bmp','png')
                is_allowed = lambda x: x.split(".")[1] in ALLOWED_EXTENSIONS
                allowed, forbidden = partition(is_allowed, files)
                list(allowed) # ['bar.exe', 'text.txt', 'data.bin']
                list(forbidden) # ['foo.jpg', 'baz.gif']


            # get unique
                df[df["gender"] == "M"]["name"].nunique() # Unique names for male

                result["Asset"].unique().tolist()

                from collections import Counter
                def filter_non_unique(lst):
                    return [item for item, count in counter = Counter(lst).items() if count == 1]
                filter_non_unique([1, 2, 2, 3, 4, 4, 5]) # [1, 3, 5]

			# remove duplicates
                li = [3, 2, 2, 1, 1, 1]
                list(set(li)) #=> [1, 2, 3]


        # Select columns
            SEC_Fill[SEC_Fill.columns[:-2]] # Show all cols except last 2

            # Select columns by dtype
                df.select_dtypes(include = "number") # Select numerical columns
                df.select_dtypes(include = "object") # Select string columns
                df.select_dtypes(include = ["datetime", "timedelta"]) # Select datetime columns
                df.select_dtypes(include = ["int8", "int16", "int32", "int64", "float"]) # Select by passing the dtypes you need


        # Doing operations with pipes:
            df_preped = (diamonds.pipe(drop_duplicates).
                                  pipe(remove_outliers, ['price', 'carat', 'depth']).
                                  pipe(encode_categoricals, ['cut', 'color', 'clarity'])
            )


		AAPL[['datetime','Volume']].head()
        market_data_250 = market_data.iloc[:250] # Select the first 250 rows
    	SPY_TICK.sample(frac=0.001)
    	pd.concat([UC.sample(n=10), UC.sample(n=10)])

        # Show rows with N/As
            events_df[events_df.isna().any(axis=1)]


		# largest
			AAPL.nlargest(20,'Volume')
			SPY_TICK.nlargest(20,'SIZE')
			NQ100.nsmallest(4,'share_volume')['share_volume']
			SPY_TICK.SIZE.nlargest(10)

			# Largest symbols by capitalization for every sector
			df.groupby('TRBC Economic Sector Name')['Company Market Cap'].nlargest(3)
			df.groupby('TRBC Economic Sector Name',as_index=False).apply(lambda x: x.nlargest(3, 'Company Market Cap'))

	
        # find duplicates 
            Index_const[Index_const.duplicated(subset=['symbol'],keep=False)]

            events_df[events_df.duplicated()]


		# importance of index
            df = pd.DataFrame({'foo':np.random.random(10000),'key':range(100,10100)})
            %timeit df[df.key==10099] # following code performs the lookup repeatedly and reports on the performance
            df_with_index = df.set_index(['key'])
            %timeit df_with_index.loc[10099]


		# Split a df into 2 random subsets
			df_1 = df.sample(frac = 0.7)
			df_2 = df.drop(df_1.index) # only works if the df index is unique


        # Select rows by values

            diamonds.price.idxmax() # position of the max

            USDCHF[USDCHF.Volume>200]
            AAPL[AAPL.Close - AAPL.Open>5]
            AAPL[AAPL.Open - AAPL.Close.shift()>15] # shows where the diff btw t-1 close and t > smth
            NQ100[(NQ100.share_volume>10000000) & (NQ100.lastsale<40)]['Name']
    		df[(df["M"] >= 50000) & (df["F"] >= 50000)] # names that atleast have 50,000 records for each gender

            diamonds[diamonds["price"].between(3500, 3700, inclusive="neither")].sample(5)


            qvdf=qvdf[pd.notnull(qvdf['comboaccrual'])]

            mask = df_results[pnl_col_name] > 0
	    	all_winning_trades = df_results[pnl_col_name].loc[mask] 

            # Select rows where Bid_Price>0 and Ask_Price>0 and Bid_Size>0 and Ask_Size>0
			training = training.ix[(training['Bid_Price']>0) | (training['Ask_Price']>0)]
			training = training.ix[(training['Bid_Size']>0) | (training['Ask_Size']>0)]


            impactful_data = self.data_df.loc[self.data_df['impact'] == 3].copy()


            cutoff=.95
            s=(qvdf['comboaccrual']<cutoff)&(qvdf['p_pman']<cutoff)&(qvdf['p_pfd']<cutoff)
            qvdf=qvdf[s]

            # delete rows where the date belong to rare dates
                unique_dates= all_hist_capital['date'].value_counts()
                all_hist_capital = all_hist_capital[~all_hist_capital['date'].isin(unique_dates[unique_dates < 5].index)] 




            # Select values which are above/below given limits
                def limit(arr, min_lim=None, max_lim=None):
                    min_check = lambda val: True if min_lim is None else (min_lim <= val)
                    max_check = lambda val: True if max_lim is None else (val <= max_lim)    
                    return [val for val in arr if min_check(val) and max_check(val)]
                data = pd.Series(np.random.randint(50,101,20))
                limit(data, 40,70)


        # Extract by date

				ETF_USA['Launch Date'] = pd.to_datetime(ETF_USA['Launch Date'], format='%Y-%m-%d')
				start = (dt.date.today() - dt.timedelta(days = 20)).strftime("%Y-%m-%d")
				ETF_USA[ETF_USA['Launch Date'] > start]


                # if data col is index:
                    df.loc['2014-01-01':'2014-02-01']
                    df.loc[df.index.month == 3]
                # if not index:
                    df[(df['date'] > '2013-01-01') & (df['date'] < '2013-02-01')]
                    df[df['date'].dt.dayofweek == 6]
                    df[df['date'].dt.month == 12]

                df.query('20130101 < date < 20130201')
                
                df.loc[datetime.date(year=2014,month=1,day=1):datetime.date(year=2014,month=2,day=1)]


                trading_days_df['DOM'] = np.where(trading_days_df.date.dt.to_period('M') != trading_days_df.date.shift().dt.to_period('M'), 'FDM', 
                                         np.where(trading_days_df.date.dt.to_period('M') != trading_days_df.date.shift(-1).dt.to_period('M'), 'EOM', 'No'))   # FDM - 1st day of month, EOM - end of Month


                qvdf=qvdf[pd.to_datetime(qvdf['release_date']).dt.date<last_valid_day.date()]
                qvdf=qvdf[pd.to_datetime(qvdf['end_date']).dt.date>=last_valid_day.date()-relativedelta(months=6)]

                mask = (stock_data['Date'] > start_date) & (stock_data['Date'] <= end_date) # filter our column based on a date range   
                stock_data = stock_data.loc[mask] # rebuild our dataframe

                all_hist_capital.loc[all_hist_capital['date'] > "2020"]
                df[df['date'].dt.to_period('Y') == '2019']

                df_filtered = df[df['date'].dt.strftime('%Y') == '2014']               
                df_filtered = df[df['date'].dt.strftime('%Y-%m') == '2014-01']
                df[df['date'].dt.to_period('M') == '2019-12']


                weekends_sales = daily_sales[daily_sales.index.dayofweek.isin([5, 6])] # filter weekends


                mkt_open = dt.datetime(int(year),int(month),int(d), 9, 30 )
                mkt_close = dt.datetime(int(year),int(month),int(d), 16, 00 )
                dat = data[(data.index > mkt_open) & (data.index<mkt_close)]
                
                after_60d = pd.to_datetime('today').date() + datetime.timedelta(days=60)
                df[df['date_col'] < after_60d]

                rng=pd.date_range('2019','2021',freq="BM")
                ts=pd.Series(np.random.randn(len(rng)),index=rng)
                ts["2019"]
                ts["2019-2":"2019-7"]
                ts.truncate(before="2019-2",after="2019-7") # select less than above

                # Only keep quotes at trading times
                df001 = df001.set_index('Date_Time')
                df001 = df001.between_time('9:30','16:00',include_start=True, include_end=True)


        # Extract by string
                # remove rows wher name of a company has on certain letters
                    qvdf=qvdf[~qvdf['name'].str[-2:].str.contains('LP')]
                    qvdf=qvdf[~qvdf['name'].str[-3:].str.contains('LLC')]

                df.loc[df['column_name'].isin(some_values)]
                df.loc[~df['column_name'].isin(some_values)]
                df.loc[df['B'].isin(['one','three'])]

                s=qvdf['industry_category'].isin([None,"Banking","Financial Services","Real Estate","Utilities"])
                qvdf=qvdf[~s]

                df['symbol'].isin(["F","AAPL","NVDA","MBI","TSLA"])

                mask = SPY_TICK['SALE_CONDITION'].values == 'F'
                SPY_TICK[mask]
                mask = SPY_TICK['SALE_CONDITION'] == 'F'
                SPY_TICK[mask]

                ETFs=MAIN_STOCKS.loc[MAIN_STOCKS['type'].isin(['ETF','ETC'])]

                # Selecting unique values from dataframe: the quickest is via numpy
                    df = pd.DataFrame({'Col1': ['Bob', 'Joe', 'Bill', 'Mary', 'Joe'],
                                    'Col2': ['Joe', 'Steve', 'Bob', 'Bob', 'Steve'],
                                    'Col3': np.random.random(5)})
                    np.unique(df[['Col1', 'Col2']].values) # array(['Bill', 'Bob', 'Joe', 'Mary', 'Steve'], dtype=object)
                    set(np.concatenate(df.values))

                # select rows, where symbol appears <10 times
                    all_hist_capital[all_hist_capital['symbol'].isin(unique_symb[unique_symb < 10].index)] 


                # separate df for every category
                    def seperated_dataframes(df, treatment):
                        treat_col = data[treatment] # col with the treatment
                        dframes_sep = [] # list to hold seperated dataframes 
                        for cat in categories(treat_col): # Go through all categories of the treatment
                            df = data[treat_col == cat] # select all rows that match the category        
                            dframes_sep.append(df) # append the selected dataframe
                        return dframes_sep

                    self.stocks = self.df['Symbol_Root'].unique()
                    for stock in self.stocks:
                        # Do aggregation
                        stock_rows = self.df.loc[self.df['Symbol_Root'] == stock]


		# identify outliers using 3 sigma approach

			df_ma = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std']) #calculate rolling mean and standard deviation
			df_ma.columns = df_ma.columns.droplevel() # drop multi-level index
			df_outliers = df.join(df_ma)
			df_outliers['outlier'] = [1 if (x > mu + 3 * sigma) 
										or (x < mu - 3 * sigma) else 0 
									for x, mu, sigma in zip(df_outliers.simple_rtn, 
																df_outliers['mean'], 
																df_outliers['std'])] 
			outliers = df_outliers.loc[df_outliers['outlier'] == 1, ['simple_rtn']]

            no_outlier_prices = prices_only[(np.abs(stats.zscore(prices_only)) < 6).all(axis=1)]

		# Filter only the largest categories
			df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")
			df.columns = map(str.lower, list(df.columns)) # convert headers to lower type
			top_genre = df["genre"].value_counts().to_frame()[0:3].index # select top 3 genre
			df_top = df[df["genre"].isin(top_genre)] # now let's filter the df with the top genre
			

# Grouping ----------------------------------------------------------------------------------------------

		values=np.random.randint(0,100,5)
		bins = pd.DataFrame({'Values':values})
		bins['Group']=pd.cut(values,range(0,101,10))

		df.resample("M")["sales"].sum() # groupby by month

    	male_df = df[df["gender"] == "M"].groupby("year").sum()
		AAPL_grouped = AAPL.groupby("Volume")
        result.groupby(["GapSize_Rank","End_Month","Strategy"]).agg({'TradeReturn': 'mean'})

        # create new col "rank": rank of market cap among symbols on every date
        qvdf["rank"] = qvdf.groupby("date")["marketCap"].rank(ascending=False)

		df[df["year"] >= 2008].pivot_table(index="name", 
											columns="year", 
											values="count", 
											aggfunc=np.sum).fillna(0)

        result[result["GapSize"] > 0.01].pivot_table(index="Asset", 
                                                    columns="Strategy", 
                                                    values="TradeReturn", 
                                                    aggfunc=np.mean).fillna(0)


        df['cap_bin']= pd.qcut(df['Company Market Cap'], q = 4, labels = ["small", "medium", "big", "very big"]) # split into bins
        df.pivot_table(index='TRBC Economic Sector Name', columns='cap_bin', aggfunc='size', fill_value=0)



        labels=['Gen X', 'Boomer', 'Greatest']
        df = pd.DataFrame({'age': [2, 67, 40, 32, 4, 15, 82, 99, 26, 30, 50, 78]})
        df['age_group'] = pd.qcut(df['age'], 3, precision = 0, labels=labels) # split into 3 groups
        df['age_group'].value_counts()






		# Combine the output of an aggregation with the original df
			d = {"orderid":[1, 1, 1, 2, 2, 3, 4, 5], 
				 "item":[10, 120, 130, 200, 300, 550, 12.3, 200],
				 "salesperson":["Nico", "Carlos", "Juan", "Nico", "Nico", "Juan", "Maria", "Carlos"]}
			df = pd.DataFrame(d)
			df["total_items_sold"] = df.groupby("orderid")["item"].transform(sum) 
			df["running_total"] = df["item"].cumsum()
			df["running_total_by_person"] = df.groupby("salesperson")["item"].cumsum()

		# merge all dfs into one dfs    
			def data_array_merge(data_array): 
				merged_df = functools.reduce(lambda left,right: pd.merge(left,right,on='Date'), data_array)
				merged_df.set_index('Date', inplace=True)
				return merged_df



		# Combine the small categories into a single category named "Others"
			d = {"genre": ["A", "A", "A", "A", "A", "B", "B", "C", "D", "E", "F"]}
			df = pd.DataFrame(d)

			# 1st way
			frequencies = df["genre"].value_counts(normalize = True) # Step 1: count the frequencies
			threshold = 0.1
			small_categories = frequencies[frequencies < threshold].index # Step 2: filter the smaller categories
			df["genre"] = df["genre"].replace(small_categories, "Other") # Step 3: replace the values
			df["genre"].value_counts(normalize = True)

			# 2nd way
			top_four = df["genre"].value_counts().nlargest(4).index
			df_updated = df.where(df["genre"].isin(top_four), other = "Other")
			df_updated["genre"].value_counts()

		# Convert continuos variable to categorical (cut and qcut)
			df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")
			pd.cut(df["Metascore"], bins = [0, 25, 50, 75, 99]).head() # Using cut you can specify the bin edges
			pd.qcut(df["Metascore"], q = 3).head() # specify the number of bins
			pd.qcut(df["Metascore"], q = 4, labels = ["awful", "bad", "average", "good"]).head() # cut and qcut accept label bin size

		# most often occuring names using collection.Counter
			from collections import Counter
			cheese = ["gouda", "brie", "feta", "cream cheese", "feta", "cheddar",
					"parmesan", "parmesan", "cheddar", "mozzarella", "cheddar", "gouda",
					"parmesan", "camembert", "emmental", "camembert", "parmesan"]
			cheese_count = Counter(cheese) # Counter is just a dictionary that maps items to number of occurrences
			# use update(more_words) method to easily add more elements to counter
			print(cheese_count.most_common(3)) # Prints: [('parmesan', 4), ('cheddar', 3), ('gouda', 2)]


# Pairwise iteration btw columns of dataframe
	from scipy.stats import ttest_ind
	from itertools import combinations

	N, M = 20, 4
	A = np.random.randn(N, M) + np.arange(M)/4 # generate a random array, add a small constant to each column
	df = pd.DataFrame(A) # converts numpy array to pandas df
	pairwise_pvalues = pd.DataFrame(columns=df.columns, index=df.columns, dtype=float)
	for (label1, column1), (label2, column2) in combinations(df.items(), 2):
		pairwise_pvalues.loc[label1, label2] = ttest_ind(column1, column2)[1]
		pairwise_pvalues.loc[label2, label1] = pairwise_pvalues.loc[label1, label2]
	pairwise_pvalues.round(3)


# List comprehension examples
	color_list = ["green", "red", "blue", "yellow"]
	rgb = [color for color in color_list if color in('green', 'red', 'blue')]

	green_list = [color for color in color_list if color == 'green']
	green_list2 = []
	for color in color_list:
		if color == 'green':
			green_list2.append(color)


	color_indicator = [0 if color == 'green'else 1 if color == 'red' else 2 if color == 'blue' else 3 for color in color_list]
	color_mapping = {'green': 0, 'red': 1, 'blue':2, 'yellow':3}
	color_indicator2 = [color_mapping[color] if color in color_mapping else 'na' for color in color_list]

	word_lengths = [len(color) for color in color_list]

	color_list1 = ['green', 'red', 'blue', 'yellow']
	color_list2 = ['dark', 'bright', 'tinted', 'glowing']
	color_matrix = [[color2 + ' ' + color1 for color1 in color_list1] for color2 in color_list2]


	result = [f(x, y) for x, y in zip(df['col1'], df['col2'])] # iterating over two columns, use `zip`

	txns = [1.09, 23.56, 57.84, 4.56, 6.78]
	TAX_RATE = .08
	def get_price_with_tax(txn):
		return txn * (1 + TAX_RATE)
	final_prices = [get_price_with_tax(i) for i in txns]

	sentence = 'the rocket came back from mars'
	vowels = [i for i in sentence if i in 'aeiou'] # list
	unique_vowels = {i for i in sentence if i in 'aeiou'} # set

	original_prices = [1.25, -9.45, 10.22, 3.78, -5.92, 1.16]
	prices = [i if i > 0 else 0 for i in original_prices]


    # From Mayer book:
        # 1
            employees = {'Alice' : 100000,'Bob' : 99817,'Carol' : 122908,'Frank' : 88123,'Eve' : 93121}

            top_earners = []
            for key, val in employees.items():
                if val >= 100000:
                    top_earners.append((key,val))
            print(top_earners)

            # or:
            top_earners = [(k, v) for k, v in employees.items() if v >= 100000]

        # 2
            text = '''
            Call me Ishmael. Some years ago - never mind how long precisely - having
            little or no money in my purse, and nothing particular to interest me
            on shore, I thought I would sail about a little and see the watery part
            of the world. It is a way I have of driving off the spleen, and regulating
            the circulation. - Moby Dick'''

            w = [[x for x in line.split() if len(x)>3] for line in text.split('\n')]

        # 3
            txt = ['Lambda-Funktionen sind anonyme Funktionen.',
            'anonyme Funktionen haben keinen Namen.',
            'Funktionen sind Objekte in Python.']

            mark = map(lambda s: (True, s) if 'anonyme' in s else (False, s), txt)
            print(list(mark))

        #4
            price = [[9.9, 9.8, 9.8, 9.4, 9.5, 9.7],
                    [9.5, 9.4, 9.4, 9.3, 9.2, 9.1],
                    [8.4, 7.9, 7.9, 8.1, 8.0, 8.0],
                    [7.1, 5.9, 4.8, 4.8, 4.7, 3.9]]

            sample = [line[::2] for line in price]



# Bid prices offered by the two buyers, pA and pB. Bid sizes, sA and sB. 
    # Add a new best size column (bS) to the table, that returns the size at the best price. 
    # If the two buyers have the same price then bS is equal to sA + sB
    # See here: https://github.com/alexanu/bestSize

        # Let's assume we have only 2 buyers:

        PRICEDOMSIZE=  5  # domain size of prices
        SIZEDOMSIZE= 100
        N = 1000 * 1000

        def createTable(N):
            return pd.DataFrame({
                    'pA': np.random.randint(0, PRICEDOMSIZE, N),
                    'pB': np.random.randint(0, PRICEDOMSIZE, N),
                    'sA': np.random.randint(0, SIZEDOMSIZE, N),
                    'sB': np.random.randint(0, SIZEDOMSIZE, N)})

        t = createTable(N)


        # 1st approach: go row by row
        def bestSizeIF(pA, pB, sA, sB):
            if pA == pB:
                return sA + sB
            return sA if pA > pB else sB

        t['bS']= t.apply(lambda row: bestSizeIF(row['pA'], row['pB'], row['sA'], row['sB']), axis=1)

        # vectorize approach with True/False vector
        def bestSizeWHERE(pA, pB, sA, sB):
            p = np.array([pA, pB])
            return np.sum(np.array([sA, sB])[
                        np.where(p == np.max(p))]) # where returns list of indices containing True values

        t['bS']= t.apply(lambda row: bestSizeWHERE(row['pA'], row['pB'], row['sA'], row['sB']), axis=1)

        def bestSizeMULT(pA, pB, sA, sB):
            p = np.array([pA, pB])
            return np.sum(np.array([sA, sB]) * (p == np.max(p))) 
                # Boolean arithmetic in which True acts as one, False acts as zero

        def bestSizeMULT2(pA, pB, sA, sB):
            return sA * (pA >= pB) + sB * (pA <= pB) # another approach

        t['bS']= t.apply(lambda row: bestSizeMULT(row['pA'], row['pB'], row['sA'], row['sB']), axis=1)

