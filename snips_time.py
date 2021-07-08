
#----------------------------------------------------------------------------------------------------
# Prep

import pandas as pd
import numpy as np
import datetime as dt
import time



SPY_TICK=pd.read_csv("D:\\Data\\tick_data\\SPY_TICK_TRADE.csv")

dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')
AAPL = pd.read_csv("D:\\Data\\minute_data\\AAPL.txt", sep='\t', decimal=",", 
                    parse_dates={'datetime': ['Date', 'Time']}, date_parser=dateparse)
AAPL.set_index('datetime',inplace=True) # "inplace" make the changes in the existing df
AAPL=AAPL.drop(AAPL.columns[[-1]],axis=1) # delete last column



read_file['Date'] = pd.to_datetime(read_file['Date']+' '+read_file['Time'] # merge 2 columns
                                    , format='%d.%m.%Y %H:%M') # adding format makes converting much faster


#----------------------------------------------------------------------------------------------------


today = dt.datetime.now().strftime("%Y-%m-%d")
n_days_ago = (dt.datetime.now() - dt.timedelta(days=8)).strftime("%Y-%m-%d")

st=int(dt.datetime(2020,12,1).timestamp())
fin=int(dt.datetime(2021,1,1).timestamp())



return_var['datetime'] = pd.to_datetime(return_var['formated'])
return_var.set_index('datetime',inplace=True) # "inplace" make the changes in the existing df


str(dt.datetime.fromtimestamp(time.time()).strftime('[%H:%M:%S] '))


from time import sleep, strftime
for ETF in SPDR_ETF:
    start_time = time.clock()
    ...
    print('Done %s out of %s in %s seconds', i, len(SPDR_ETF), round(time.clock() - start_time, 2))
    print('sleeping.....')
    sleep(randint(45,60))




# Attributes of dates
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

    weekends_sales = daily_sales[daily_sales.index.dayofweek.isin([5, 6])] # filter weekends

    # df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern') # convert to Eastern Time
    df['Weekday_Name'] = df.index.dayofweek # or "weekday", 'dayofyear'
    df['Hour'] = df.index.hour # "minute"


# Resample
    AAPL.index # shows freq=None
    AAPL.asfreq('D') # H, W; important that index are datetime
    AAPL.asfreq('H').isna().any(axis=1)
    AAPL.asfreq('H', method = 'ffill')

    AAPL.resample('W').mean()
    AAPL.rolling(window = 7, center = True).mean()

    daily_sales = df.resample("D")["sales"].sum().to_frame() # agregate by day


    #convert tick data to 15 minute data
        data_frame = pd.read_csv(tick_data_file, 
                                names=['id', 'deal', 'Symbol', 'Date_Time', 'Bid', 'Ask'], 
                                index_col=3, parse_dates=True, skiprows= 1)
        ohlc_M15 =  data_frame['Bid'].resample('15Min').ohlc()
        ohlc_H1 = data_frame['Bid'].resample('1H').ohlc()
        ohlc_H4 = data_frame['Bid'].resample('4H').ohlc()
        ohlc_D = data_frame['Bid'].resample('1D').ohlc()

    def resample( data ):
        dat       = data.resample( rule='1min', how='mean').dropna()
        dat.index = dat.index.tz_localize('UTC').tz_convert('US/Eastern')
        dat       = dat.fillna(method='ffill')
        return dat


# Examples of filtering dataframe by date
    impactful_data['timestamp_af'] = impactful_data['timestamp'].apply(lambda x: self.utc_to_local(x))

    qvdf=qvdf[pd.to_datetime(qvdf['release_date']).dt.date<last_valid_day.date()]

    qvdf=qvdf[pd.to_datetime(qvdf['end_date']).dt.date>=last_valid_day.date()-relativedelta(months=6)]

    mask = (stock_data['Date'] > start_date) & (stock_data['Date'] <= end_date) # filter our column based on a date range   
    stock_data = stock_data.loc[mask] # rebuild our dataframe

    df['CohortIndex_d'] = (df['last_active_date'] - df['signup_date']).dt.days # new column with the difference between the two dates


# Create date ranges
	pd.date_range('2016-08-01','2017-08-01')
	dates = pd.date_range('2016-08-01','2017-08-01', freq='M')
	idx = pd.date_range("2018-1-1",periods=20,freq="Q")
	ts = pd.Series(range(len(idx)),index=idx)
	ts.resample("2H").mean()

    
    [int(x.timestamp()) for x in pd.date_range('2016-08-01','2017-08-01', freq='M')]


    [print(x,y) for x,y in zip(pd.date_range("2018-01-06", periods=10,freq="30d"),pd.date_range("2018-02-05", periods=10,freq="30d"))]

	pd.Series(range(10),index=pd.date_range("2000",freq="D",periods=10))

    start=pd.Timestamp("2018-01-06 00:00:00")
    pd.date_range("2018-01-06", periods=10,freq="30d")

	dates = pd.date_range('2016-08-01','2017-08-01', freq='M')
    dates.shift(1)



    rng=pd.date_range(start,end,freq="D")
    ts=pd.Series(np.random.randn(len(rng)),index=rng)
    ts.shift(2)[1]

# dummy datasets with dates
	import pandas.util.testing as tm
	tm.N, tm.K = 5,3
	tm.makeTimedeltaIndex(), tm.makeTimeSeries(), tm.makePeriodSeries()
	tm.makeDateIndex(), tm.makePeriodIndex(), tm.makeObjectSeries()


	from itertools import product
	datecols = ['year', 'month', 'day']
	df = pd.DataFrame(list(product([2016,2017],[1,2],[1,2,3])),columns = datecols)
	df['data']=np.random.randn(len(df))
	df.index = pd.to_datetime(df[datecols])


# Align 2 datetime series and interpolate if necessary

    from datetime import date, time
    import numpy as np
    from ..errors import MqValueError

    def __interpolate_step(x: pd.Series, dates: pd.Series = None) -> pd.Series:
        if x.empty:
            raise MqValueError('Cannot perform step interpolation on an empty series')

        first_date = pd.Timestamp(dates.index[0]) if isinstance(x.index[0], pd.Timestamp) else dates.index[0]

        # locate previous valid date or take first value from series
        prev = x.index[0] if first_date < x.index[0] else x.index[x.index.get_loc(first_date, 'pad')]

        current = x[prev]

        curve = x.align(dates, 'right', )[0]                  # only need values from dates

        for knot in curve.iteritems():
            if np.isnan(knot[1]):
                curve[knot[0]] = current
            else:
                current = knot[1]
        return curve

    def interpolate(x: pd.Series, dates: Union[List[date], List[time], pd.Series] = None,
                    method: Interpolate = Interpolate.INTERSECT) -> pd.Series:
        """
        Interpolate over specified dates or times
        Stepwize interpolation of series based on dates in second series:

        >>> a = generate_series(100)
        >>> b = generate_series(100)
        >>> interpolate(a, b, Interpolate.INTERSECT)

        """
        if dates is None:
            dates = x

        if isinstance(dates, pd.Series):
            align_series = dates
        else:
            align_series = pd.Series(np.nan, dates)

        if method == Interpolate.INTERSECT: # Only returns a value for valid dates
            return x.align(align_series, 'inner')[0]
        if method == Interpolate.NAN: # Value will be NaN for dates not present in the series
            return x.align(align_series, 'right')[0]
        if method == Interpolate.ZERO: # Value will be zero for dates not present in the series
            align_series = pd.Series(0.0, dates)
            return x.align(align_series, 'right', fill_value=0)[0]
        if method == Interpolate.STEP: # Value of the previous valid point
            return __interpolate_step(x, align_series)
        else:
            raise MqValueError('Unknown intersection type: ' + method)

    def align(x: Union[pd.Series, Real], 
            y: Union[pd.Series, Real], 
            method: Interpolate = Interpolate.INTERSECT) -> \
            Union[List[pd.Series], List[Real]]:
    
        """
        Align dates of two series or scalars
        Stepwize interpolation of series based on dates in second series:

        >>> a = generate_series(100)
        >>> b = generate_series(100)
        >>> align(a)

        """
        if isinstance(x, Real) and isinstance(y, Real):
            return [x, y]
        if isinstance(x, Real):
            return [pd.Series(x, index=y.index), y]
        if isinstance(y, Real):
            return [x, pd.Series(y, index=x.index)]

        if method == Interpolate.INTERSECT: # Resultant series only have values on the intersection of dates /times
            return x.align(y, 'inner')
        if method == Interpolate.NAN: # Values will be NaN for dates or times only present in the other series
            return x.align(y, 'outer')
        if method == Interpolate.ZERO: # Values will be zero for dates or times only present in the other series
            return x.align(y, 'outer', fill_value=0)
        if method == Interpolate.TIME: # Missing values surrounded by valid values will be interpolated
            new_x, new_y = x.align(y, 'outer')
            new_x.interpolate('time', limit_area='inside', inplace=True)
            new_y.interpolate('time', limit_area='inside', inplace=True)
            return [new_x, new_y]
        if method == Interpolate.STEP: # use the value of the previous valid point
            new_x, new_y = x.align(y, 'outer')
            new_x.fillna(method='ffill', inplace=True)
            new_y.fillna(method='ffill', inplace=True)
            new_x.fillna(method='bfill', inplace=True)
            new_y.fillna(method='bfill', inplace=True)
            return [new_x, new_y]
        else:
            raise MqValueError('Unknown intersection type: ' + method)


# -------------------------------------------------------------------------------------------------------------
# Business days and biz hours
	start = dt.datetime(2019,1,1), end = dt.datetime(2019,10,10)
	pd.date_range(start,end)
	pd.bdate_range(start,end,freq=10)
    pd.bdate_range(start,end,freq='BM')
    
	pd.bdate_range(start,periods=4,freq="BQS")


	rng=pd.date_range('2019','2021',freq="BM")
	ts=pd.Series(np.random.randn(len(rng)),index=rng)
	ts["2018"]
	ts["2019-2":"2019-7"]
	ts.truncate(before="2019-2",after="2019-7") # select less than above

	# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components

	two_biz_days=2*pd.offsets.BDay()
	friday = pd.Timestamp("2018-01-05")
	friday.day_name()
	two_biz_days.apply(friday).day_name()
	(friday+two_biz_days),(friday+two_biz_days).day_name()

	ts =pd.Timestamp("2018-01-06 00:00:00")
	ts.day_name() # --> "Saturday"
	offset=pd.offsets.BusinessHour(start="09:00")
	offset.rollforward(ts) # Bring the date to the closest offset date (Monday)

	pd.offsets.BusinessHour() # from 9 till 17
	rng = pd.date_range("2018-01-10","2018-01-15",freq="BH") # BH is "business hour"
	rng+pd.DateOffset(months=2,hours=3)

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




    # biz days btw 2 dates

        # http://dateutil.readthedocs.io/en/stable/rrule.html
        from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR
        def daterange(start_date, end_date):
            # automate a range of business days between two dates
            return rrule(DAILY, dtstart=start_date, until=end_date, byweekday=(MO,TU,WE,TH,FR))

        for tr_date in daterange(start_date, end_date):

    # itertools through biz days

        now   = dt.date.today()
        year  = str(now.year)
        m     = str(now.month)
        month = '0'+m

        day_5 = now - 5 * BDay()
        day_4 = now - 4 * BDay()
        day_3 = now - 3 * BDay()
        day_2 = now - 2 * BDay()
        day_1 = now - 1 * BDay()
        day_0 = now - 0 * BDay() # Add current day_0

        days  = [day_5.day, day_4.day, day_3.day, day_2.day, day_1.day, day_0.day]
        months = [day_5.month, day_4.month, day_3.month, day_2.month, day_1.month, day_0.month]
        years = [day_5.year, day_4.year, day_3.year, day_2.year, day_1.year, day_0.year]
        days  = [str(d) for d in days]
        months  = [str(ms) for ms in months]
        years  = [str(ys) for ys in years]
            
        # for day in days:
            for (day, month, year) in itertools.izip(days, months, years):
                try:

    # closest biz day in the past

        def closest_business_day_in_past(date=None):
            if date is None:
                date = dt.datetime.today()
            return date + BDay(1) - BDay(1)


        date = pd.datetime.strptime(pd.datetime.now().strftime('%Y%m%d'),'%Y%m%d')  
                - pd.offsets.BDay(1)

# trading calendars examples

    # https://github.com/alexanu/market_calendars
    # Chinese and US trading calendars with date math utilities 
    # based on pandas_market_calendar
    # Speed is achieved via Cython

    import market_calendars as mcal
    from market_calendars.core import Date, Period, Calendar, Schedule

    cal_sse = mcal.get_calendar('China.SSE') # create chinese shanghai stock exchange calendar
    cal_nyse = mcal.get_calendar('NYSE') # create nyse calendar
    cal_sse.name, cal_sse.tz # return name and time zone => ('China.SSE', <DstTzInfo 'Asia/Shanghai' LMT+8:06:00 STD>)

    cal_sse.holidays('2018-09-20', '2018-10-10') # return holidays in datetime format
    cal_sse.holidays('2018-09-20', '2018-10-10', return_string=True) # return holidays in string format
    cal_sse.holidays('2018-09-20', '2018-10-10', return_string=True, include_weekends=False) # return holidays excluding weekends
    cal_sse.biz_days('2015-05-20', '2015-06-01') # return biz days in datetime format

    cal_sse.is_biz_day('2014-09-22'), cal_sse.is_biz_day('20140130') # => (True, True)
    cal_sse.is_holiday('2016-10-01'), cal_sse.is_holiday('2014/9/21') # => (True, True)
    cal_sse.is_weekend('2014-01-25'), cal_sse.is_weekend('2011/12/31') # => (True, True)
    cal_sse.is_end_of_month('2011-12-30'), cal_sse.is_end_of_month('20120131') # => (True, True)



    cal_sse.advance_date('2017-04-27', '2b') # => datetime.datetime(2017, 5, 2, 0, 0)
    cal_sse.advance_date('20170427', '2b', return_string=True) # => '2017-05-02'
    cal_sse.advance_date('20170427', '1w', return_string=True) # => '2017-05-04'
    cal_sse.advance_date('20170427', '1m', return_string=True) # => '2017-05-31'
    cal_sse.advance_date('20170427', '-1m', return_string=True) # => '2017-03-27'

    cal_sse.schedule('2018-01-05', '2018-02-01', '1w', return_string=True, date_generation_rule=2) # => ['2018-01-05', '2018-01-12', '2018-01-19', '2018-01-26', '2018-02-01']

# Holiday detection for US Markets
    import datetime as dt
    import pandas.tseries.holiday as pd_holiday

    class USTradingCalendar(pd_holiday.AbstractHolidayCalendar):
        rules = [
            pd_holiday.Holiday('NewYearsDay',month=1,day=1,observance=pd_holiday.nearest_workday),
            pd_holiday.USMartinLutherKingJr,
            pd_holiday.USPresidentsDay,
            pd_holiday.GoodFriday,
            pd_holiday.USMemorialDay,
            pd_holiday.Holiday('USIndependenceDay',month=7,day=4,observance=pd_holiday.nearest_workday),
            pd_holiday.USLaborDay,
            pd_holiday.USThanksgivingDay,
            pd_holiday.Holiday('Christmas',month=12,day=25,observance=pd_holiday.nearest_workday)
        ]

    def get_trading_close_holidays(year=None):
        use_year = year
        if not use_year:
            use_year = int(dt.datetime.utcnow().year)
        inst = USTradingCalendar()
        return inst.holidays(
            dt.datetime(use_year-1, 12, 31),
            dt.datetime(use_year, 12, 31))

    def is_holiday(date=None,date_str=None,fmt='%Y-%m-%d'):
        cal_df = None
        use_date = dt.datetime.utcnow()
        if date:
            use_date = date
        else:
            if date_str:
                use_date = dt.datetime.strptime(date_str,fmt)
        cal_df = get_trading_close_holidays(year=use_date.year)
        use_date_str = use_date.strftime(fmt)
        for d in cal_df.to_list():
            if d.strftime(fmt) == use_date_str:
                return True
        return False

# list of weekdays in a particular year-month
    import calendar
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

# Is it holiday for markets?
    from holidays import UnitedStates
        def is_trading_day(self, timestamp):
            """Tests whether markets are open on a given day."""

            # Markets are closed on holidays.
            if timestamp in UnitedStates():
                self.logs.debug("Identified holiday: %s" % timestamp)
                return False

            # Markets are closed on weekends.
            if timestamp.weekday() in [5, 6]:
                self.logs.debug("Identified weekend: %s" % timestamp)
                return False

            # Otherwise markets are open.
            return True

# several time ranges during trading session

    start_trading_day = datetime.date(2017, 6, 8)
    end_trading_day = datetime.date(2017, 6, 20) #end_trading_day

    start =  start or datetime.date(1900,1,1)
    stop = stop or datetime.date.today()


    # end_trading_day  = datetime.date.today()
    trading_days = []
    while start_trading_day <= end_trading_day:
        trading_days.append(start_trading_day)
        start_trading_day += datetime.timedelta(days=1)

    for trading_d in trading_days:
        start_datetime1 = datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 9, 0, 0)
        end_datetime1 =   datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 10, 15, 0)
        start_datetime2 = datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 10, 30, 0)
        end_datetime2 =   datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 11, 30, 0)
        start_datetime3 = datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 13, 30, 0)
        end_datetime3 =   datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 15, 0, 0)
        time_query = {
            '$or':[{'snapshot_time':{'$gte':start_datetime1, '$lte':end_datetime1}},
                {'snapshot_time':{'$gte':start_datetime2, '$lte':end_datetime2}},
                {'snapshot_time':{'$gte':start_datetime3, '$lte':end_datetime3}}
                ]}

# Selecting only trading hours

    def trading_start(d):
        mkt_open = dt.datetime( int(year), int(month), int(d), 9, 30 )
        return mkt_open

    def trading_end(d):
        mkt_close = dt.datetime( int(year), int(month), int(d), 16, 00 )
        return mkt_close

    def trading_hours(data):
        test = []
        for d in days:
            dat = data[ ( data.index > trading_start(d) ) & ( data.index < trading_end(d) ) ]
            test.append( dat )
        return test

    # Only keep quotes at trading times
    df001 = df001.set_index('Date_Time')
    df001 = df001.between_time('9:30','16:00',include_start=True, include_end=True)

# check if market is open

    import logging
    import os
    import inspect
    import sys
    import time
    import pytz
    from enum import Enum
    from datetime import datetime
    from govuk_bank_holidays.bank_holidays import BankHolidays

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    from Utility.Utils import Utils


    class TimeAmount(Enum):
        """Types of amount of time to wait for
        """

        SECONDS = 0
        NEXT_MARKET_OPENING = 1

class TimeProvider:

    def __init__(self):
        logging.debug("TimeProvider __init__")

    def is_market_open(self, timezone):
        """
        Return True if the market is open, false otherwise
            - **timezone**: string representing the timezone
        """
        tz = pytz.timezone(timezone)
        now_time = datetime.now(tz=tz).strftime("%H:%M")
        return BankHolidays().is_work_day(datetime.now(tz=tz)) and Utils.is_between(
            str(now_time), ("07:55", "16:35")
        )

    def get_seconds_to_market_opening(self, from_time):
        """Return the amount of seconds from now to the next market opening,
        taking into account UK bank holidays and weekends"""
        today_opening = datetime(
            year=from_time.year,
            month=from_time.month,
            day=from_time.day,
            hour=8,
            minute=0,
            second=0,
            microsecond=0,
        )

        if from_time < today_opening and BankHolidays().is_work_day(from_time.date()):
            nextMarketOpening = today_opening
        else:
            # Get next working day
            nextWorkDate = BankHolidays().get_next_work_day(date=from_time.date())
            nextMarketOpening = datetime(
                year=nextWorkDate.year,
                month=nextWorkDate.month,
                day=nextWorkDate.day,
                hour=8,
                minute=0,
                second=0,
                microsecond=0,
            )
        # Calculate the delta from from_time to the next market opening
        return (nextMarketOpening - from_time).total_seconds()

    def wait_for(self, time_amount_type, amount=-1):
        """Wait for the specified amount of time.
        An TimeAmount type can be specified
        """
        if time_amount_type is TimeAmount.NEXT_MARKET_OPENING:
            amount = self.get_seconds_to_market_opening(datetime.now())
        elif time_amount_type is TimeAmount.SECONDS:
            if amount < 0:
                raise ValueError("Invalid amount of time to wait for")
        logging.info("Wait for {0:.2f} hours...".format(amount / 3600))
        time.sleep(amount)


# -------------------------------------------------------------------------------------------------------------    
# string to datetime -> calculate -> date string for d days ago
# today is August 13, 10 days ago means August 3

    def days_ago(d, start_date=None):
        if start_date==None:
            date = datetime.datetime.today() - datetime.timedelta(days=d)
        else:
            date = str_to_date(start_date) - datetime.timedelta(days=d)
        return date.strftime("%Y-%m-%d")

    def str_to_date(dt):
        year, month, day = (int(x) for x in dt.split('-'))    
        return datetime.date(year, month, day)

# transform to the valid data:
    # value "09/2007" to date 2007-09-01. 
    # value "2006" to date 2016-01-01

    def parse_thisdate(text: str) -> datetime.date:
        parts = text.split('/')
        if len(parts) == 2:
            return datetime.date(int(parts[1]), int(parts[0]), 1)
        elif len(parts) == 1:
            return datetime.date(int(parts[0]), 1, 1)
        else:
            assert False, 'Unknown date format'

# creating datetime object from zip

    with zipfile.ZipFile(self.fname) as zf:
        with zf.open(zf.namelist()[0]) as infile:
            header = infile.readline()
            datestr, record_count = header.split(b':')
            self.month = int(datestr[2:4])
            self.day = int(datestr[4:6])
            self.year = int(datestr[6:10])
            utc_base_time = datetime(self.year, self.month, self.day)
            self.base_time = timezone('US/Eastern').localize(utc_base_time).timestamp()

# creating url which contains date

    date = pd.datetime.today() - pd.offsets.BDay(1)
    date=date.date()
    datestr = date.strftime('%Y%m%d')
    daystr=str(date.day)
    monthstr=str(date.month)
    yearstr=str(date.year)
    
    
    url='http://www.netfonds.no/quotes/exchange.php?'  
    url=url+'exchange=%s'
    url=url+'&at_day=' + daystr
    url=url+'&at_month=' +monthstr
    url=url+'&at_year=' +yearstr
    url=url+'&format=csv'

    date_str = f'{date.year}{date.month:02d}{date.day:02d}'
    reddit_request = requests.get(f'http://archive.org/wayback/available?url=reddit.com/r/{term}&timestamp={date_str}')

# date to string to date

    # date to string
    current_date = Date(2015, 7, 24) # create date object
    two_days_later = current_date + 2 # => Date(2015, 7, 26) 
    str(two_days_later) # => 2015-07-26
    current_date + '1M' # => Date(2015, 8, 24)
    current_date + Period('1M') # same with previous line # => Date(2015, 8, 24)

    current_date.strftime("%Y%m%d") # => '20150724'
    Date.strptime('20160115', '%Y%m%d') # => Date(2016, 1, 15)
    Date.strptime('2016-01-15', '%Y-%m-%d') # => Date(2016, 1, 15)
    Date.from_datetime(datetime.datetime(2015, 7, 24)) # => Date(2015, 7, 24)

    # string to date
    pd.to_datetime(pd.Series(["Jul 31, 2017","2010-10-01","2016/10/10","2014.06.10"]))
    pd.to_datetime(pd.Series(["11 Jul 2018","13.04.2015","30/12/2011"]),dayfirst=True)
    # providing a format could increase speed of conversion significantly
    pd.to_datetime(pd.Series(["12-11-2010 01:56","11-01-2012 22:10","28-02-2013 14:59"]), format='%d-%m-%Y %H:%M')

    import market_calendars as mcal
    cal_sse = mcal.get_calendar('China.SSE') # create chinese shanghai stock exchange calendar
    cal_sse.adjust_date('20130131') # => datetime.datetime(2013, 1, 31, 0, 0)
    cal_sse.adjust_date('20130131', return_string=True) # => '2013-01-31'
    cal_sse.adjust_date('2017/10/01') # => datetime.datetime(2017, 10, 9, 0, 0)
    cal_sse.adjust_date('2017/10/01', convention=2) # => datetime.datetime(2017, 9, 29, 0, 0)

# reading time stamp
    def read_hhmmss(field: str) -> int:
        # """Read a HH:MM:SS field and return us since midnight.""
        if field != "":
            hour = int(field[0:2])
            minute = int(field[3:5])
            second = int(field[6:8])
            return 1000000 * ((3600 * hour) + (60 * minute) + second)
        else:
            return 0


    def read_hhmmssmil(field: str) -> int:
        """Read a HH:MM:SS:MILL field and return us since midnight."""
        if field != "":
            hour = int(field[0:2])
            minute = int(field[3:5])
            second = int(field[6:8])
            msecs = int(field[9:])
            return ((1000000 * ((3600 * hour) + (60 * minute) + second)) +
                    (1000 * msecs))
        else:
            return 0


    def read_mmddccyy(field: str) -> np.datetime64:
        """Read a MM-DD-CCYY field and return a np.datetime64('D') type."""
        if field != "":
            month = int(field[0:2])
            day = int(field[3:5])
            year = int(field[6:10])
            return np.datetime64(
                datetime.date(year=year, month=month, day=day), 'D')
        else:
            return np.datetime64(datetime.date(year=1, month=1, day=1), 'D')


    def read_ccyymmdd(field: str) -> np.datetime64:
        """Read a CCYYMMDD field and return a np.datetime64('D') type."""
        if field != "":
            year = int(field[0:4])
            month = int(field[4:6])
            day = int(field[6:8])
            return np.datetime64(
                datetime.date(year=year, month=month, day=day), 'D')
        else:
            return np.datetime64(datetime.date(year=1, month=1, day=1), 'D')


    def read_timestamp_msg(dt_tm: str) -> Tuple[np.datetime64, int]:
        """Read a CCYYMMDD HH:MM:SS field."""
        if dt_tm != "":
            (date_str, time_str) = dt_tm.split(' ')
            dt = read_ccyymmdd(date_str)
            tm = read_hhmmss(time_str)
            return dt, tm
        else:
            return np.datetime64(datetime.date(year=1, month=1, day=1), 'D'), 0


    def read_hist_news_timestamp(dt_tm: str) -> Tuple[np.datetime64, int]:
        """Read a news story time"""
        if dt_tm != "":
            date_str = dt_tm[0:8]
            time_str = dt_tm[8:14]
            dt = read_ccyymmdd(date_str)
            tm = read_hhmmss_no_colon(time_str)
            return dt, tm
        else:
            return np.datetime64(datetime.date(year=1, month=1, day=1), 'D'), 0

# named granularity into seconds

    def granularity_to_time(s):
        """convert a named granularity into seconds.
        get value in seconds for named granularities: M1, M5 ... H1 etc.
        >>> print(granularity_to_time("M5"))
        300
        """
        mfact = {
            'S': 1,
            'M': 60,
            'H': 3600,
            'D': 86400,
            'W': 604800,
        }
        try:
            f, n = re.match("(?P<f>[SMHDW])(?:(?P<n>\d+)|)", s).groups()
            n = n if n else 1
            return mfact[f] * int(n)

        except Exception as e:
            raise ValueError(e)

# -------------------------------------------------------------------------------------------------------------
# UTC convertion and timezone
    from pytz import timezone
    from pytz import utc

    # We're using NYSE and NASDAQ, which are both in the easters timezone.
    MARKET_TIMEZONE = timezone("US/Eastern")

    def utc_to_market_time(self, timestamp):
        """Converts a UTC timestamp to local market time."""

        utc_time = utc.localize(timestamp)
        market_time = utc_time.astimezone(MARKET_TIMEZONE)

        return market_time

    def market_time_to_utc(self, timestamp):
        """Converts a timestamp in local market time to UTC."""

        market_time = MARKET_TIMEZONE.localize(timestamp)
        utc_time = market_time.astimezone(utc)

        return utc_time

    def as_market_time(self, year, month, day, hour=0, minute=0, second=0):
        """Creates a timestamp in market time."""

        market_time = datetime(year, month, day, hour, minute, second)
        return MARKET_TIMEZONE.localize(market_time)


    # Remove timezone information.
    def unlocalize(dateTime):
        return dateTime.replace(tzinfo=None)


    def localize(dateTime, timeZone):
        """Returns a datetime adjusted to a timezone:

        * If dateTime is a naive datetime (datetime with no timezone information), timezone information is added but date
        and time remains the same.
        * If dateTime is not a naive datetime, a datetime object with new tzinfo attribute is returned, adjusting the date
        and time data so the result is the same UTC time.
        """

        if datetime_is_naive(dateTime):
            ret = timeZone.localize(dateTime)
        else:
            ret = dateTime.astimezone(timeZone)
        return ret


    def as_utc(dateTime):
        return localize(dateTime, pytz.utc)


    def datetime_to_timestamp(dateTime):
        """ Converts a datetime.datetime to a UTC timestamp."""
        diff = as_utc(dateTime) - epoch_utc
        return diff.total_seconds()


    def timestamp_to_datetime(timeStamp, localized=True):
        """ Converts a UTC timestamp to a datetime.datetime."""
        ret = datetime.datetime.utcfromtimestamp(timeStamp)
        if localized:
            ret = localize(ret, pytz.utc)
        return ret


    epoch_utc = as_utc(datetime.datetime(1970, 1, 1))

    def utc_to_local(self, utc_dt):
        utc_dt = datetime.strptime(utc_dt, "%Y-%m-%d %H:%M:%S")
    local = utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
    return local.strftime("%Y-%m-%d %H:%M:%S")

    def utc_to_local(utc_dt):
        local = utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
        return local.strftime("%Y-%m-%d %H:%M:%S")

    utc_dt = datetime.strptime("2018-11-01 01:45:00", "%Y-%m-%d %H:%M:%S")
    print(utc_to_local(utc_dt))

# -------------------------------------------------------------------------------------------------------------
# Adjust dates in csv according to the time zone

        time_zone_difference = int(args[2])
        input_filename = args[1]
        output_filename = 'OUT_' + input_filename
        with open(output_filename, 'w') as w:
            with open(input_filename, 'r') as r:
                reader = csv.reader(r, delimiter=';')
                for row in reader:
                    print(row)
                    new_row = list(row)
                    ts = datetime.strptime(new_row[0], '%Y%m%d %H%M%S')
                    ts += timedelta(hours=time_zone_difference)
                    new_row[0] = ts.strftime('%Y%m%d %H%M%S')
                    w.write(';'.join(new_row) + '\n')

# ------------------------------------------------------------------------------------------

def is_third_friday(day=None):
    """ check if day is month's 3rd friday """
    day = day if day is not None else datetime.datetime.now()
    defacto_friday = (day.weekday() == 4) or (
        day.weekday() == 3 and day.hour() >= 17)
    return defacto_friday and 14 < day.day < 22

def after_third_friday(day=None):
    """ check if day is after month's 3rd friday """
    day = day if day is not None else datetime.datetime.now()
    now = day.replace(day=1, hour=16, minute=0, second=0, microsecond=0)
    now += relativedelta.relativedelta(weeks=2, weekday=relativedelta.FR)
    return day > now

def get_first_monday(year):
    ret = datetime.date(year, 1, 1)
    if ret.weekday() != 0:
        diff = 7 - ret.weekday()
        ret = ret + datetime.timedelta(days=diff)
    return ret

def get_last_monday(year):
    ret = datetime.date(year, 12, 31)
    if ret.weekday() != 0:
        diff = ret.weekday() * -1
        ret = ret + datetime.timedelta(days=diff)
    return ret

# ------------------------------------------------------------------------------------------

def promt_time_stamp():
    return str(datetime.datetime.fromtimestamp(time.time()).strftime('[%H:%M:%S] '))
print(promt_time_stamp() + 'load dowjones tickers ..')



#-----------------------------------------------------------------------------------------------------------------------

class Period:
    """Periods available for the charts
    Values are the minutes of each period
    """
    m1 = 1
    m5 = 5
    m15 = 15
    m30 = 30
    h1 = 60  # 1 hour
    h4 = 240  # 4 hours
    d1 = 1440  # 1 day
    w1 = 10080  # 1 week
    mn1 = 43200 # 1 month

#-----------------------------------------------------------------------------------------------------------------------       
#-----------------------------------------------------------------------------------------------------------------------

# Time Enum
class Span(Enum):
    FIVE_MINUTE = "5minute"   # Five minute's time
    TEN_MINUTE = "10minute"   # Ten minute's time
    DAY = "day"               # 24 hours' time
    WEEK = "week"             # 7 days' time
    YEAR = "year"             # 365 days' time


def get_history(self, symbol, interval = Span.DAY, span = Span.YEAR, bounds = Bounds.REGULAR):
        return self.trader.get_historical_quotes(symbol, interval.value, span.value, bounds.value)



# day of the week

    from datetime import datetime

    def ask_month_day_year(prompt: str = "Enter your birthday"):
        date = input(f"{prompt} in the format: MM/DD/YYYY ")
        month, day, year = (int(x.strip()) for x in date.split("/"))
        return month, day, year

    def day_of_the_week(year: int, month: int, day: int) -> str:
        return f"{datetime(year, month, day):%A}"

    month, day, year = ask_month_day_year()
    print(f"You were born on a {day_of_the_week(year, month, day)}.")

# -----------------------------------------------------------------------------------------------
# Convert daily dataframe into intraday start times

    def set_index_to_intraday_start(daily_df):
        
        """Convert daily dataframe into intraday start times.
        For example, the "2020-02-02" session will be indexed to "2020-02-01 18:00"
        Args:
            daily_df (frame): daily indicator with timestamp index in YYYY-MM-DD format in EST
        Returns:
            frame: index set to intraday start time in EST. You will need to ffill
                the result if merging into an intraday dataframe
        Example:
            >>> ppframe_daily = indicator_pp_daily(df_ins)["2020-02-02":]
            >>> ppframe = set_index_to_intraday_start(ppframe_daily)
            >>> df_ins[ppframe.columns] = ppframe
            >>> df_ins[ppframe.columns] = df_ins[ppframe.columns].fillna(method="ffill")
            If you want to go the other way, you can do:
            >>> my_dates = list(df.index.date)
            >>> session_end = tradingday_offset(
            >>>     start=my_dates[0],
            >>>     end=my_dates[-1],
            >>>     dates=my_dates,
            >>>     offsets=1,
            >>>     roll="forward",
            >>> ).tolist()
            >>> df.index = pd.to_datetime(session_end)
        """
        df = daily_df.copy()
        df.index = pd.to_datetime(df.index)
        df["stop"] = df.index + pd.to_timedelta(17, "h")
        df["start"] = df.stop - pd.to_timedelta(23, "h")
        df.set_index("start", inplace=True)
        df.drop(["stop"], axis=1, inplace=True)
        return df

