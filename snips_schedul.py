
import os	
import pandas.io.data as web
import datetime as dt
from apscheduler.schedulers.blocking import BlockingScheduler

def download_SPX():
    s_dir = os.getcwd() + '/' + date.strftime('%Y-%m-%d')
    start = dt.datetime(1990,1,1)
    f = web.DataReader("^GSPC", 'yahoo', start, dt.date.today()).fillna('NaN')
    f.to_csv(s_dir+'/SPX.csv', date_format='%Y%m%d')	
	
def main():
    sched = BlockingScheduler()
    @sched.scheduled_job('cron', day_of_week='mon,tue,wed,thu,fri,sat', hour=23)
    def scheduled_job():
    	print('[INFO] Job started.')
        download_SPX()
        print('[INFO] Job ended.')

    sched.start()    

if __name__ == '__main__':
	main()
	
	
	
# -----------------------------------------------------------------------------------------------------------------------------

# Scheduler imports
from apscheduler.schedulers.blocking import BlockingScheduler

# Job imports
import main as job
from emails import email_login

def bob_job():

    me, password = email_login()
    sched = BlockingScheduler()

    @sched.scheduled_job('cron', day_of_week='mon,tue,wed,thu,fri', hour=17)
    def scheduled_job():
        job.run(me, password)

    sched.start()    


if __name__ == '__main__':
    bob_job()


#-----------------------------------------------------------------------------------------------
...

lMessages = []

for symbol in aSymbols:
    ...
    if not after_opening_rang_bars.empty:
        if symbol not in existing_order_symbols:
            if not after_opening_rang_breakout.empty:
                ...
                lMessages.append(f"Placing order for {symbol} at {limit_price}, closed_above {opening_range_high} \n\n {after_opening_rang_breakout.iloc[0]}")
                print(f"Placing order for {symbol} at {limit_price}, closed_above {opening_range_high} at {after_opening_rang_breakout.iloc[0]}")
                ...

        else:
            print(f"Already an order for {symbol}, skipping")

print(EMAIL_ADDRESS)
print(EMAIL_PASSWORD)
with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

    email_message = f"Subject: Trade Notifications for {current_date}\n\n"
    email_message += "\n".join(lMessages)

    server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, email_message)

#--------------------------------------------------------------------------------------------------------------


