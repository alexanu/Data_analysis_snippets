# https://github.com/agespido/SkyScraper/blob/main/src/sky.py



"""
Html: 
<span class="BpkText_bpk-text__ZWIzZ BpkText_bpk-text--caption__NDJhY Price_totalPrice__MTJhN">Gesamt 2.318 €</span>

Selector:
#app-root > div.FlightsDayView_row__NjQyZ > div > div > div > div:nth-child(1) > div.FlightsResults_dayViewItems__ZDFlO > div:nth-child(1) > div > div > a > div > div.BpkTicket_bpk-ticket__paper__ZTQxN.BpkTicket_bpk-ticket__stub__Y2M3M.Ticket_stub__NGYxN.BpkTicket_bpk-ticket__stub--padded__ZTlkM.BpkTicket_bpk-ticket__stub--horizontal__YjRhZ > div > div > span

JS Path: document.querySelector("#app-root > div.FlightsDayView_row__NjQyZ > div > div > div > div:nth-child(1) > div.FlightsResults_dayViewItems__ZDFlO > div:nth-child(1) > div > div > a > div > div.BpkTicket_bpk-ticket__paper__ZTQxN.BpkTicket_bpk-ticket__stub__Y2M3M.Ticket_stub__NGYxN.BpkTicket_bpk-ticket__stub--padded__ZTlkM.BpkTicket_bpk-ticket__stub--horizontal__YjRhZ > div > div > span")

XPath:
//*[@id="app-root"]/div[1]/div/div/div/div[1]/div[3]/div[1]/div/div/a/div/div[2]/div/div/span

Full XPath:
/html/body/div[3]/div[4]/div/div[2]/div[1]/div/div/div/div[1]/div[3]/div[1]/div/div/a/div/div[2]/div/div/span

"""


domain_codes = ['de','nl','uk','ua'] # domain
destination_codes = ['sana'] # destination
departure_dates = ['230404','230405'] # dep_date
return_dates = ['230414','230415'] # ret_date

url = f'https://www.skyscanner.{domain}/transport/fluge  /muc/{destination}/{dep_date}/{ret_date}/?adults=2&adultsv2=2&cabinclass=economy&children=1&childrenv2=7%7c1&duration=1320&inboundaltsenabled=false&infants=1&outboundaltsenabled=false&preferdirects=false&ref=home&rtn=1'
        https://www.skyscanner.de      /transport/flights/muc/sana         /230404/230414        /?adultsv2=2&cabinclass=economy&childrenv2=7%7c3&duration=1320&inboundaltsenabled=false&outboundaltsenabled=false&rtn=1&stops=!oneStop,!twoPlusStops



from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import geckodriver_autoinstaller
import warnings 
from time import sleep
import random
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

airport_dict = {
	'sana': 'SANDIEGO',
}

class Trip:
	def __init__(self, _departure_ap, _arrival_ap, _date):
		self.departure_ap = _departure_ap
		self.arrival_ap = _arrival_ap
		self.date = _date
		self.price = ''
		self.dep_time = ''
		self.arr_time = ''
		self.duration = ''
		self.connections = ''

	def __str__(self):
		return '\033[92mFlight {} -> {} on {} at {} (duration {}) => {}\033[0m'.format(airport_dict[self.departure_ap], airport_dict[self.arrival_ap], self.date, self.dep_time, self.duration, self.price)

	def get_data(self, driver):
		wait(4)
		WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="dayview-first-result"]/div/div[3]/div/div/div/span')))
		self.price = driver.find_element(By.XPATH, '//*[@id="dayview-first-result"]/div/div[3]/div/div/div/span').text
		self.dep_time = driver.find_element(By.XPATH, '//*[@id="dayview-first-result"]/div/div[1]/div/div[1]/div[3]/div/div[2]/div[1]/span[1]/div/span').text
		self.duration = driver.find_element(By.XPATH, '//*[@id="dayview-first-result"]/div/div[1]/div/div[1]/div[3]/div/div[2]/div[2]/span').text
		self.arr_time = driver.find_element(By.XPATH, '//*[@id="dayview-first-result"]/div/div[1]/div/div[1]/div[3]/div/div[2]/div[3]/span[1]/div/span[1]').text
		self.connections = driver.find_element(By.XPATH, '//*[@id="dayview-first-result"]/div/div[1]/div/div[1]/div[3]/div/div[2]/div[2]/div[2]/span').text

	def write_to_file(self):
		fname = 'travel_all_over_the_world.csv'
		# Write file header if file does not exist
		if not os.path.isfile(fname):
			f = open(fname, 'w')
			f.write('Date, Departure Airport, Departure Time, Arrival Airport, Arrival Time, Flight Duration, Number of Connections, Price\n')
		else:
			f = open(fname, 'a')
		f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(self.date, airport_dict[self.departure_ap], self.dep_time, airport_dict[self.arrival_ap], self.arr_time, self.duration, self.connections, self.price))
		f.close()
def set_url(base_url, departure_airport, arrival_airport, oneway_date, payload):
	return '{}/{}/{}/{}{}'.format(base_url, departure_airport, arrival_airport, oneway_date, payload)

def wait(min = 2, max = 4):
	if max <= min:
		max = min + 1
	sleep(random.uniform(min, max))

def increase_date(date):
	# if the last two characters are greater than 31, increase the first two characters by 1
	if int(date[-2:]) >= 31:
		date = str(int((int(date) + 100) / 100) * 100) # Add 1 month
	if int(date[-4:-2]) >= 12:
		date = str(int((int(date) + 10000) / 10000) * 10000 + 1000) # Add 1 year
	return str(int(date) + 1)

def init_driver():
	warnings.filterwarnings("ignore", category=DeprecationWarning) # Ignore deprecation warnings
	geckodriver_autoinstaller.install()
	options = Options()
	options.headless = True
	cap = DesiredCapabilities().FIREFOX
	cap["marionette"] = True
	driver = webdriver.Firefox(capabilities=cap, options=options)
	return driver

DEBUG = False # Must be False for production

# Fields for the URL
BASE_URL = 'https://www.skyscanner.es'
URL = BASE_URL + '/transporte/vuelos'
DEPARTURE_AIRPORT = 'muc'
ARRIVAL_AIRPORTS = list(airport_dict.keys())[1:]
EARLIEST_DATE = '230404' # Use as a first date to iterate over
PAYLOAD = '?adultsv2=2&cabinclass=economy&childrenv2=7%7c3&duration=1320&inboundaltsenabled=false&outboundaltsenabled=false&rtn=1&stops=!oneStop,!twoPlusStops' # Other fields

# Other variables
LATEST_BACK_DATE = '230414'
MAX_ATTEMPTS = 3

def get_trips(driver, dep_ap, arr_ap, dep_date=EARLIEST_DATE, back_date=LATEST_BACK_DATE):
	trip = Trip(dep_ap, arr_ap, dep_date)
	attempts = 0
	while(int(trip.date) < int(back_date)):
		try:
			url = set_url(URL, trip.departure_ap, trip.arrival_ap, trip.date, PAYLOAD)
			driver.get(url)
			trip.get_data(driver)
			print(trip.__str__())
		except Exception as e:
			attempts += 1
			print('\033[91mError: could not get information for date {} (attempt {}/{})\033[0m'.format(trip.date, attempts, MAX_ATTEMPTS))
			print(e) if DEBUG else None
			if attempts >= MAX_ATTEMPTS:
				print('Skipping date {}'.format(trip.date))
				trip.date = increase_date(trip.date)
				attempts = 0
			driver.quit()
			wait(1, 10)
			driver = init_driver()
			continue

		trip.write_to_file()
		trip.date = increase_date(trip.date)
		attempts = 0
		wait()

def main():
	# One-way trips
	for arr_airport in ARRIVAL_AIRPORTS:
		driver = init_driver()
		get_trips(driver, DEPARTURE_AIRPORT, arr_airport)
		driver.quit()
	
	# Back trips
	for arr_airport in ARRIVAL_AIRPORTS:
		driver = init_driver()
		get_trips(driver, arr_airport, DEPARTURE_AIRPORT)
		driver.quit()

if __name__ == '__main__':
	main()