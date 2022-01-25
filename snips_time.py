
#----------------------------------------------------------------------------------------------------
# Prep

import pandas as pd
import numpy as np

import datetime as dt
import time
import calendar



dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
AAPL = pd.read_csv("D:\\Data\\minute_data\\US\\Stocks_adj\\AAPL.txt", sep=',', decimal=".", 
                    parse_dates=['datetime'], date_parser=dateparse)
AAPL.set_index('datetime',inplace=True) # "inplace" make the changes in the existing df

AAPL.head()
AAPL.info()

len(AAPL)


#---------------------------------------------------------------------------------------------------




df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5], 'hour': [10,11]})
pd.to_datetime(df[['month','day','year']])
pd.to_datetime(df)




# Dates/times representation, conversion ---------------------------------------------------------

    df = pd.DataFrame({'date': [1470195805, 1480195805, 1490195805], 'value': [2, 3, 4]})
    pd.to_datetime(df['date'], unit='s')
    df['date'].astype('datetime64[s]')

    st=int(dt.datetime(2020,12,1).timestamp())
    fin=int(dt.datetime(2021,1,1).timestamp())
    [int(x.timestamp()) for x in pd.date_range('2016-08-01','2017-08-01', freq='M')]

    # date to string
        current_date = Date(2015, 7, 24) # create date object
        two_days_later = current_date + 2 # => Date(2015, 7, 26) 
        str(two_days_later) # => 2015-07-26
        current_date + '1M' # => Date(2015, 8, 24)
        current_date + Period('1M') # same with previous line # => Date(2015, 8, 24)
        current_date.strftime("%Y%m%d") # => '20150724'

        str(dt.datetime.fromtimestamp(time.time()).strftime('[%H:%M:%S] '))

    # string to date
        pd.to_datetime(pd.Series(["Jul 31, 2017","2010-10-01","2016/10/10","2014.06.10"]))
        pd.to_datetime(pd.Series(["11 Jul 2018","13.04.2015","30/12/2011"]),dayfirst=True)

        df = pd.DataFrame({'date': ['3/10/2000', 'a/11/2000', '3/12/2000'], 'value': [2, 3, 4]})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.set_index('date',inplace=True) # "inplace" make the changes in the existing df

        # providing a format could increase speed of conversion significantly
        pd.to_datetime(pd.Series(["12-11-2010 01:56","11-01-2012 22:10","28-02-2013 14:59"]), format='%d-%m-%Y %H:%M')


        Date.strptime('20160115', '%Y%m%d') # => Date(2016, 1, 15)
        Date.strptime('2016-01-15', '%Y-%m-%d') # => Date(2016, 1, 15)

        friday = pd.Timestamp("2018-01-05")
        friday.day_name()

        df = pd.DataFrame({'date_start': ['3/10/2000', '3/11/2000', '3/12/2000'],
                           'date_end': ['3/11/2000', '3/12/2000', '3/13/2000'],
                           'value': [2, 3, 4]})
        df = df.astype({'date_start': 'datetime64','date_end': 'datetime64'})

        # transform to the valid data:
            # value "09/2007" to date 2007-09-01. 
            # value "2006" to date 2016-01-01
            '''
            def parse_thisdate(text: str) -> dt.date:
                parts = text.split('/')
                if len(parts) == 2:
                    return dt.date(int(parts[1]), int(parts[0]), 1)
                elif len(parts) == 1:
                    return dt.date(int(parts[0]), 1, 1)
                else:
                    assert False, 'Unknown date format'
            '''
        # Read a MM-DD-CCYY field and return a np.datetime64('D') type
            '''
            def read_mmddccyy(field: str) -> np.datetime64:
                if field != "":
                    month = int(field[0:2])
                    day = int(field[3:5])
                    year = int(field[6:10])
                    return np.datetime64(dt.date(year=year, month=month, day=day), 'D')
                else:
                    return np.datetime64(dt.date(year=1, month=1, day=1), 'D')
            '''

    # string to datetime -> calculate -> date string for d days ago
        year, month, day = (int(x) for x in dt.split('-'))
        date = dt.date(year, month, day) - dt.timedelta(days=d)
        date.strftime("%Y-%m-%d")


# Timezones --------------------------------------------------------------------------------

    from pytz import timezone
    from pytz import utc

    # Remove timezone information.
    def unlocalize(dateTime):
        return dateTime.replace(tzinfo=None)

    # Add timezone information
    def localize(dateTime, timeZone):
        if dt.datetime_is_naive(dateTime): # If dateTime is a naive datetime (datetime with no timezone information), ...
            ret = timeZone.localize(dateTime) # ... timezone information is added
        else: # If dateTime is not a naive datetime ... 
            ret = dateTime.astimezone(timeZone)
        return ret


    MARKET_TIMEZONE = timezone("US/Eastern") # NYSE, NASDAQ

    now_time = dt.datetime.now(tz=MARKET_TIMEZONE).strftime("%H:%M")

    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern') # convert to Eastern Time

    def utc_to_local(utc_dt):
        utc_dt = dt.strptime(utc_dt, "%Y-%m-%d %H:%M:%S")
        local = utc_dt.replace(tzinfo=utc).astimezone(tz=None)
        return local.strftime("%Y-%m-%d %H:%M:%S")

    print(utc_to_local(utc_dt))

    def utc_to_market_time(timestamp):
        """Converts a UTC timestamp to local market time."""
        utc_time = utc.localize(timestamp)
        market_time = utc_time.astimezone(MARKET_TIMEZONE)
        return market_time

    impactful_data['timestamp_af'] = impactful_data['timestamp'].apply(lambda x: utc_to_market_time(x))

    def market_time_to_utc(timestamp):
        """Converts a timestamp in local market time to UTC."""
        market_time = MARKET_TIMEZONE.localize(timestamp)
        utc_time = market_time.astimezone(utc)
        return utc_time


# Array of dates & hours ---------------------------------------------------------------------------

	pd.date_range('2016-08-01','2017-08-01')
	dates = pd.date_range('2016-08-01','2017-08-01', freq='M')
    dates.shift(1)

	idx = pd.date_range("2018-1-1",periods=20,freq="Q")
	ts = pd.Series(range(len(idx)),index=idx)
    pd.Series(range(10),index=pd.date_range("2000",freq="D",periods=10))

    [int(x.timestamp()) for x in pd.date_range('2016-08-01','2017-08-01', freq='M')]

    # Paired periods:
    [print(x,y) for x,y in zip(pd.date_range("2018-01-06", periods=10,freq="30d"),pd.date_range("2018-02-05", periods=10,freq="30d"))]


    # Biz dates range
        start = dt.datetime(2019,1,1)
        end = dt.datetime(2019,10,10)
        pd.bdate_range(start,end,freq=10)
        pd.bdate_range(start,end,freq='BM')
        pd.date_range('2019','2021',freq="BM") 
        pd.bdate_range(start,periods=4,freq="BQS")
            # be careful: bdate_range or BDay() are just calendar days with weekends stripped out (ie. it doesn't take holidays into account).


    # biz days btw 2 dates
        start_date = dt.datetime(2019,1,1)
        end_date = dt.datetime(2019,10,10)
        # http://dateutil.readthedocs.io/en/stable/rrule.html
        from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR
        def daterange(start_date, end_date):
            # automate a range of business days between two dates
            return rrule(DAILY, dtstart=start_date, until=end_date, byweekday=(MO,TU,WE,TH,FR))

        for tr_date in daterange(start_date, end_date):
            print(tr_date)


    # Biz hours range
        pd.offsets.BusinessHour() # from 9 till 17
        rng = pd.date_range("2018-01-10","2018-01-15",freq="BH") # BH is "business hour"
        rng+pd.DateOffset(months=2,hours=3)
            # be careful: bdate_range or BDay() are just calendar days with weekends stripped out (ie. it doesn't take holidays into account).


    def month_weekdays(year_int, month_int):
        """
        Produces a list of datetime.date objects representing the
        weekdays in a particular month, given a year.
        """
        cal = calendar.Calendar()
        return [
            d for d in cal.itermonthdates(year_int, month_int)
            if d.weekday() < 5 and d.year == year_int
        ]
    month_weekdays(2020,4)


	from itertools import product
	datecols = ['year', 'month', 'day']
	df = pd.DataFrame(list(product([2016,2017],[1,2],[1,2,3])),columns = datecols)
	df['data']=np.random.randn(len(df))
	df.index = pd.to_datetime(df[datecols])


    # dummy datasets with dates
        import pandas.util.testing as tm
        tm.N, tm.K = 5,3
        tm.makeTimedeltaIndex(), tm.makeTimeSeries(), tm.makePeriodSeries()
        tm.makeDateIndex(), tm.makePeriodIndex(), tm.makeObjectSeries()


# Today vs days_ago, periods between, how long smth took  -----------------------------------

    today = dt.datetime.now().strftime("%Y-%m-%d")
    n_days_ago = (dt.datetime.now() - dt.timedelta(days=8)).strftime("%Y-%m-%d")

    start = dt.date.today() - dt.timedelta(days = 365*2)
    end = dt.date.today()

    current_date = Date(2015, 7, 24) # create date object
    two_days_later = current_date + 2 # => Date(2015, 7, 26) 
    current_date + '1M' # => Date(2015, 8, 24)

    import dateutil.relativedelta
    MONTH_CUTTOFF = 5
    currentDate = dt.date.today() + dt.timedelta(days=1)
    pastDate = currentDate - dateutil.relativedelta.relativedelta(months=MONTH_CUTTOFF)

    # string to datetime -> calculate -> date string for d days ago
        year, month, day = (int(x) for x in dt.split('-'))
        date = dt.date(year, month, day) - dt.timedelta(days=d)
        date.strftime("%Y-%m-%d")


    def days_between(self, d1, d2):
        d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)


    for ETF in SPDR_ETF:
        start_time = time.time()
        ...
        print('Done %s out of %s in %s seconds', i, len(SPDR_ETF), round(time.time() - start_time, 2))
        print('sleeping.....')
        time.sleep(randint(45,60))

    df['CohortIndex_d'] = (df['last_active_date'] - df['signup_date']).dt.days # new column with the difference between the two dates

    # Biz days offset
        two_biz_days=2*pd.offsets.BDay()
        friday = pd.Timestamp("2018-01-05")
        friday.day_name()
        two_biz_days.apply(friday).day_name()
        (friday+two_biz_days),(friday+two_biz_days).day_name()

        now   = dt.date.today()
        day_5 = now - 5 * pd.offsets.BDay()

            # be careful: bdate_range or BDay() are just calendar days with weekends stripped out (ie. it doesn't take holidays into account).

    # Offset to biz hours
        ts =pd.Timestamp("2018-01-06 00:00:00")
        ts.day_name() # --> "Saturday"
        offset=pd.offsets.BusinessHour(start="09:00")
        offset.rollforward(ts) # Bring the date to the closest offset date (Monday)


    df.index = pd.to_datetime(df.index)
    df["stop"] = df.index + pd.to_timedelta(17, "h")
    df["start"] = df.stop - pd.to_timedelta(23, "h")
    df.set_index("start", inplace=True)
    df.drop(["stop"], axis=1, inplace=True)


    # closest biz day in the past
    date = pd.datetime.strptime(pd.datetime.now().strftime('%Y%m%d'),'%Y%m%d') - pd.offsets.BDay(1)


# Attributes of dates ---------------------------------------------------------------------------
    AAPL.head(10)

    dir(AAPL.index)

    AAPL['Year'] = AAPL.index.year
    AAPL['Month'] = AAPL.index.month # 'quarter'
    AAPL['Month'] = AAPL.index.month_name
    AAPL['Week'] = AAPL.index.weekofyear # 'week'
    AAPL['Weekday_Name'] = AAPL.index.dayofweek # or "weekday", 'dayofyear'
    AAPL['Hour'] = AAPL.index.hour # "minute"
    AAPL['Days_in_Mo'] = AAPL.index.daysinmonth # "days_in_month"
    AAPL[AAPL.index.is_month_start] # 'is_month_end', 'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start'

    '''
    Here it would be useful to have other attributes:
        NORMWE (Last_working_day_before_normal_we)
        LONGWE (Last_working_day_before_long_we)
        EOM	(Last_working_day_in_month)
        EOQ	(Last_working_day_in_quarter)
    '''
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components


    weekends_sales = daily_sales[daily_sales.index.dayofweek.isin([5, 6])] # filter weekends

	friday = pd.Timestamp("2018-01-05")
	friday.day_name()

    btc_data = alpaca.get_crypto_bars('BTCUSD', TimeFrame.Day, "2021-02-08", "2021-10-18").df
    btc_data.index = btc_data.index.map(lambda timestamp : timestamp.date) # keep only the date part of our timestamp index



# Resample / group
    AAPL.index # shows freq=None
    AAPL.asfreq('D') # H, W; important that index are datetime
    AAPL.asfreq('H').isna().any(axis=1)
    AAPL.asfreq('H', method = 'ffill')

    AAPL.resample('W').mean()
    AAPL.resample("2H").mean()
    AAPL.rolling(window = 7, center = True).mean()

    daily_trade_volumes = AAPL.resample("D")["Volume"].sum().to_frame() # agregate by day


    #convert tick data to 15 minute data
        data_frame = pd.read_csv(tick_data_file, 
                                names=['id', 'deal', 'Symbol', 'Date_Time', 'Bid', 'Ask'], 
                                index_col=3, parse_dates=True, skiprows= 1)
        ohlc_M15 =  data_frame['Bid'].resample('15Min').ohlc()
        ohlc_H1 = data_frame['Bid'].resample('1H').ohlc()
        ohlc_H4 = data_frame['Bid'].resample('4H').ohlc()
        ohlc_D = data_frame['Bid'].resample('1D').ohlc()

        # The same with minute data:
        AAPL_ohlc_M15 =  AAPL['Close'].resample('15Min').ohlc()

    def resample( data ):
        dat       = data.resample( rule='1min', how='mean').dropna()
        dat.index = dat.index.tz_localize('UTC').tz_convert('US/Eastern')
        dat       = dat.fillna(method='ffill')
        return dat

    # Resample example: 1Min into 5Min
    data_5m = {}
    for ticker in data_dump:
        logic = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
        data_5m[ticker] = data_dump[ticker].resample('5Min').apply(logic)
        data_5m[ticker].dropna(inplace=True)



# Filter df by date -----------------------------------------------------------------------------------------

    
    qvdf=qvdf[pd.to_datetime(qvdf['release_date']).dt.date<last_valid_day.date()]
    qvdf=qvdf[pd.to_datetime(qvdf['end_date']).dt.date>=last_valid_day.date()-relativedelta(months=6)]

    mask = (stock_data['Date'] > start_date) & (stock_data['Date'] <= end_date) # filter our column based on a date range   
    stock_data = stock_data.loc[mask] # rebuild our dataframe

    all_hist_capital.loc[all_hist_capital['date'] > "2020"]

    weekends_sales = daily_sales[daily_sales.index.dayofweek.isin([5, 6])] # filter weekends


    mkt_open = dt.datetime(int(year),int(month),int(d), 9, 30 )
    mkt_close = dt.datetime(int(year),int(month),int(d), 16, 00 )
    dat = data[(data.index > mkt_open) & (data.index<mkt_close)]
    

    rng=pd.date_range('2019','2021',freq="BM")
	ts=pd.Series(np.random.randn(len(rng)),index=rng)
	ts["2019"]
	ts["2019-2":"2019-7"]
	ts.truncate(before="2019-2",after="2019-7") # select less than above

    # Only keep quotes at trading times
    df001 = df001.set_index('Date_Time')
    df001 = df001.between_time('9:30','16:00',include_start=True, include_end=True)


# trading calendars examples

    # https://github.com/rsheftel/pandas_market_calendars
    # Chinese and US trading calendars with date math utilities 
    # based on pandas_market_calendar
    # Speed is achieved via Cython
    # 
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar('NYSE') 
    nyse_extract = nyse.schedule(start_date='2020-12-01', end_date='2021-11-01')
    mcal.date_range(nyse_extract, frequency='1M')
    nyse.valid_days(start_date='2020-12-01', end_date='2021-11-01')



    import pandas_market_calendars as mcal
    print(mcal.get_calendar_names()) # Show available calendars

    nyse = mcal.get_calendar('NYSE') 
    nyse.tz.zone
    nyse.open_time, nyse.close_time
    nyse.open_time_on("1950-01-01"), 
    nyse.get_time_on("market_close", "1960-01-01")
    nyse.get_time("post"), nyse.get_time("pre")

    nyse_extract = nyse.schedule(start_date='2019-07-01', end_date='2022-12-31')
    nyse_extract_extended = nyse.schedule(start_date='2019-07-01', end_date='2022-12-31', start="pre", end="post") # including pre and post-market


    mcal.date_range(nyse_extract, frequency='1H')

    holidays = nyse.holidays()
    holidays.holidays[:10]

    nyse.valid_days(start_date='2016-12-20', end_date='2017-01-10')
    nyse.schedule(start_date='2012-07-01', end_date='2012-07-10') # shows early close
    nyse.early_closes(schedule = nyse_extract)
    nyse.early_closes(schedule = nyse_extract_extended)
    
    nyse.open_at_time(nyse_extract, pd.Timestamp('2020-07-02 12:00', tz='America/New_York'))
    nyse.open_at_time(nyse_extract, pd.Timestamp('2020-07-02 17:00', tz='America/New_York'))
    nyse.open_at_time(nyse_extract_extended, pd.Timestamp('2020-07-02 17:00', tz='America/New_York'))

    print(nyse.regular_market_times) # more on this under the 'Customizations' heading

    lse = mcal.get_calendar('LSE')
    lse_extract = lse.schedule(start_date='2019-07-01', end_date='2022-12-31')
    mcal.merge_schedules(schedules=[nyse_extract, lse_extract], how='inner') # dates where both the NYSE and LSE are open

    alpaca.get_calendar("2021-02-08", "2021-02-18") # start=None, end=None



# Market open?
    def pre_market_open():
        pre_market_start_time = dt.datetime.now().replace(hour=12, minute=00, second=00, tzinfo=utc).timestamp()
        market_start_time = dt.datetime.now().replace(hour=13, minute=30, second=00, tzinfo=utc).timestamp()
        right_now = dt.datetime.now().replace(tzinfo=utc).timestamp()
        if market_start_time >= right_now >= pre_market_start_time:
            return True
        else:
            return False

    def post_market_open():
        post_market_end_time = dt.datetime.now().replace(hour=22, minute=30, second=00, tzinfo=timezone.utc).timestamp()
        market_end_time = dt.datetime.now().replace(hour=20, minute=00, second=00, tzinfo=timezone.utc).timestamp()
        right_now = dt.datetime.now().replace(tzinfo=timezone.utc).timestamp()
        if post_market_end_time >= right_now >= market_end_time:
            return True
        else:
            return False

    def regular_market_open():
        market_start_time = dt.datetime.now().replace(hour=13, minute=30, second=00, tzinfo=timezone.utc).timestamp()
        market_end_time = dt.datetime.now().replace(hour=20, minute=00, second=00, tzinfo=timezone.utc).timestamp()
        right_now = dt.datetime.now().replace(tzinfo=timezone.utc).timestamp()
        if market_end_time >= right_now >= market_start_time:
            return True
        else:
            return False