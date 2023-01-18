
# HTTP Trigger---------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Example 0

            # Content of "__init__.py"
                import logging
                import azure.functions as func

                def main(req: func.HttpRequest) -> func.HttpResponse:
                    logging.info('Python HTTP trigger function processed a request.')

                    name = req.params.get('name')
                    if not name:
                        try:
                            req_body = req.get_json()
                        except ValueError:
                            pass
                        else:
                            name = req_body.get('name')

                    if name:
                        return func.HttpResponse(f"Hello {name}!")
                    else:
                        return func.HttpResponse(
                            "Please pass a name on the query string or in the request body",
                            status_code=400
                        )

            # Related "function.json":
                {
                    "scriptFile": "__init__.py",
                    "disabled": false,    
                    "bindings": [
                        {
                            "authLevel": "function", #  function-specific API key is required
                            "type": "httpTrigger",
                            "direction": "in",
                            "name": "req"
                        },
                        {
                            "type": "http",
                            "direction": "out",
                            "name": "$return"
                        }
                    ]
                }




        # Example 1
                # Content of "__init__.py":
                    import logging
                    import azure.functions as func

                    def main(req: func.HttpRequest) -> func.HttpResponse:
                        logging.info('Python HTTP trigger function processed a request.')

                        name = req.params.get('name')
                        if not name:
                            try:
                                req_body = req.get_json()
                            except ValueError:
                                pass
                            else:
                                name = req_body.get('name')

                        if name:
                            return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
                        else:
                            return func.HttpResponse(
                                "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
                                status_code=200
                            )

                # Related "function.json" :
                    {
                    "scriptFile": "__init__.py",
                    "bindings": [
                        {
                        "authLevel": "anonymous",
                        "type": "httpTrigger",
                        "direction": "in",
                        "name": "req",
                        "methods": [
                            "get",
                            "post"
                        ]
                        },
                        {
                        "type": "http",
                        "direction": "out",
                        "name": "$return"
                        }
                    ]
                    }

        # Example 2
            # Content of "__init__.py":
                import logging
                import azure.functions as func

                def main(request: func.HttpRequest):
                    logging.info(f'Requested method: {request.method}')

                    if request.method == "GET":
                        return func.HttpResponse("Hello!")
                    elif request.method == "POST":
                        try:
                            name = request.get_json()['name']
                        except (ValueError, KeyError):
                            return func.HttpResponse("Wrong JSon")        
                        return func.HttpResponse(f'Hello, {name}')

            # Related "function.json" :
                {
                    "scriptFile": "__init__.py",
                    "bindings": [
                        {
                            "name": "request",
                            "direction": "in",
                            "type": "httpTrigger",
                            "authLevel": "anonymous",
                            "methods": ["get", "post"]
                        },
                        {
                            "direction": "out",
                            "type": "http",
                            "name": "$return"
                        }
                    ]
                }

        # Example 3
                # Content of "__init__.py":
                        import logging
                        import os
                        import pyodbc
                        import json
                        import azure.functions as func

                        def main(req: func.HttpRequest) -> func.HttpResponse:
                            logging.info('Python HTTP trigger function processed a request.')

                            locId = req.params.get('locId')
                            if not locId:
                                try:
                                    req_body = req.get_json()
                                except ValueError:
                                    pass
                                else:
                                    locId = req_body.get('locId')

                            sql = "SELECT * FROM ssc.schedule_settings WHERE locId=" + locId + " ORDER BY shiftId;"

                            conn = pyodbc.connect(os.environ['DMCP_CONNECT_STRING'])
                            cursor = conn.cursor()
                            cursor.execute(sql)
                            rows = cursor.fetchall()
                            cursor.close()
                            conn.close()

                            shifts = {}

                            for row in rows:
                                if not row[7] in shifts.keys():
                                    shifts[row[7]] = {}
                                    shifts[row[7]]['openShifts'] = row[2]
                                    shifts[row[7]]['bartenders'] = row[4]
                                    shifts[row[7]]['locId'] = row[10]

                            if shifts:
                                return func.HttpResponse(json.dumps(shifts))

                # Related "function.json" :
                        {
                        "scriptFile": "__init__.py",
                        "bindings": [
                            {
                            "authLevel": "function",
                            "type": "httpTrigger",
                            "direction": "in",
                            "name": "req",
                            "methods": [
                                "get",
                                "post"
                            ]
                            },
                            {
                            "type": "http",
                            "direction": "out",
                            "name": "$return"
                            }
                        ]
                        }

        # Example 4
                # Content of "__init__.py":

                    import azure.functions as func
                    import pymongo
                    import json
                    from bson.objectid import ObjectId
                    import logging

                    def main(req: func.HttpRequest) -> func.HttpResponse:

                        # example call http://localhost:7071/api/getAdvertisement/?id=5eb6cb8884f10e06dc6a2084

                        id = req.params.get('id')
                        print("--------------->", id)
                        
                        if id:
                            try:
                                url = "mongodb://cmsdatabasenp:5qHfQ.....Name=@cmsdatabasenp@"  # TODO1: Update with appropriate MongoDB connection information
                                client = pymongo.MongoClient(url)
                                database = client['azure']
                                collection = database['advertisements']
                            
                                query = {'_id': ObjectId(id)}
                                result = collection.find_one(query)
                                print("----------result--------")

                                result = json.dumps(result)
                                print(result)

                                return func.HttpResponse(result, mimetype="application/json", charset='utf-8')
                            except:
                                return func.HttpResponse("Database connection error.", status_code=500)

                        else:
                            return func.HttpResponse("Please pass an id parameter in the query string.", status_code=400)


                # Related "function.json" :
                    {
                    "scriptFile": "__init__.py",
                    "bindings": [
                        {
                        "authLevel": "Anonymous",
                        "type": "httpTrigger",
                        "direction": "in",
                        "name": "req",
                        "methods": [
                            "get",
                            "post"
                            ]
                        },
                        {
                        "type": "http",
                        "direction": "out",
                        "name": "$return"
                        }
                    ]
                    }


        # Example 5

            import logging, os, csv, urllib.parse, json
            from typing import List, Optional
            from datetime import datetime, timedelta, date
            from time import time

            from requests import *
            import azure.functions as func

            from azure.storage.blob import BlobServiceClient, ContainerClient
            from azure.identity import DefaultAzureCredential

            from shared.helpers import *


            def main(req: func.HttpRequest) -> func.HttpResponse:
                logging.info("Python HTTP trigger function received a request.")
                req_body = req.get_json()
                start_date = date.fromisoformat(req_body.get("start_date", (date.today() - timedelta(days=1)).isoformat()))
                end_date = date.fromisoformat(req_body.get("end_date", date.today().isoformat()))
                county = req_body.get("county", "hays")
                judicial_officers = req_body.get("judicial_officers", [])
                ms_wait = int(req_body.get("ms_wait", "200"))
                location = req_body.get("location", None)

                # Call scraper with given parameters
                scrape(start_date,end_date, ..., location)

                print("Returning response...")
                return func.HttpResponse(
                    f"Finished scraping cases for {judicial_officers} in {county} from {start_date} to {end_date}",
                    status_code=200,
                )


            # The scraper itself
            def scrape(
                start_date: date,
                county: str,
                judicial_officers: List[str],
                ms_wait: int,
                location: Optional[str],
                ):

                # initializer blob container client for sending html files to
                blob_connection_str = os.getenv("AzureWebJobsStorage")
                container_name_html = os.getenv("blob_container_name_html")
                blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(blob_connection_str)
                container_client = blob_service_client.get_container_client(container_name_html)

                session = requests.Session()
                ...

                # initialize variables to time script and build a list of already scraped cases
                START_TIME = time()

                # loop through each day
                for date in (start_date + timedelta(n) for n in range((end_date - start_date).days + 1)):
                    date_string = datetime.strftime(date, "%m/%d/%Y")
                    # Need underscore since azure treats slashes as new files
                    date_string_underscore = datetime.strftime(date, "%m_%d_%Y")

                    # loop through each judicial officer
                    for JO_name in judicial_officers:
                        if JO_name not in judicial_officer_to_ID:
                            logger.error(
                                f"judicial officer {JO_name} not found on search page. Continuing."
                            )
                            continue
                        JO_id = judicial_officer_to_ID[JO_name]
                        logger.info(f"Searching cases on {date_string} for {JO_name}")
                        # POST a request for search results

                                # write case html data
                                logger.info(f"{len(case_html)} response string length")
                                # write to blob
                                blob_name = f"{file_hash_dict['case_no']}:{county}:{date_string_underscore}:{file_hash_dict['file_hash']}.html"
                                logger.info(f"Sending {blob_name} to blob...")
                                write_string_to_blob(file_contents=case_html, blob_name=blob_name, container_client=container_client, container_name=container_name_html)
                                if test:
                                    logger.info("Testing, stopping after first case")
                                    return

                logger.info(f"\nTime to run script: {round(time() - START_TIME, 2)} seconds")

            def write_to_blob(file, case_id):
                blob_connection_str = os.getenv("blob_connection_str")
                blob_container_name = os.getenv("blob_container_name")
                blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(
                    blob_connection_str
                )
                container = blob_service_client.get_container_client(blob_container_name)

                with open(file, "rb") as data:
                    container.upload_blob(name=case_id, data=data)


# Time Trigger---------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Example 0
            # Content of "__init__.py":
                    import datetime
                    import logging
                    import azure.functions as func

                    def main(mytimer: func.TimerRequest) -> None:
                        utc_timestamp = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

                        if mytimer.past_due:
                            logging.info('The timer is past due!')

                        logging.info('Python timer trigger function ran at %s', utc_timestamp)

            # Related "function.json" :

                    # cron expression:
                    # The pattern we use to represent every 5 minutes is 0 */5 * * * *. 
                    # This, in plain text, means: "When seconds is equal to 0, 
                    #                               minutes is divisible by 5, 
                    #                               for any hour, day of the month, month, day of the week, or year".

                    # “0 30 0 1 * *” - first day of each month at 0:30

                    {
                    "scriptFile": "__init__.py",
                    "bindings": [
                        {
                        "name": "mytimer",
                        "type": "timerTrigger",
                        "direction": "in",
                        "schedule": "0 */1 * * * *"
                        }
                    ]
                    }

        # Example 2: picks up the raw CSV files from our Azure Blob Store container, does transformations and loads the data onto SQL database

            # Content of "__init__.py"
                import os
                import pandas as pd
                import azure.functions as func
                from azure.storage.blob import ContainerClient, BlobClient
                import pyodbc

                SERVER = os.environ.get("DB_SERVER")
                DB_NAME = os.environ.get("DB_NAME")
                USERNAME = os.environ.get("DB_USERNAME")
                PASSWORD = os.environ.get("DB_PASSWORD")
                CONN_STRING = os.environ.get("CONN_STRING")
                CONTAINER_NAME = 'data'
                TABLE_NAME = 'transactions'


                def get_data(conn_str: str, container_name: str) -> pd.DataFrame:
                    """Fetches transactions csv files from Blob store, read into Pandas DataFrames and concatenate them all into one DataFrame """
                    container = ContainerClient.from_connection_string(conn_str=conn_str,container_name = container_name)
                    blob_list = container.list_blobs()
                    if not os.path.exists('./data'):
                        os.makedirs('./data')

                    dfs = []
                    for blob in blob_list:
                        blob_client = BlobClient.from_connection_string(conn_str=conn_str,container_name=container_name, blob_name=blob.name)
                        with open(blob.name, "wb") as my_blob:
                                blob_data = blob_client.download_blob()
                                blob_data.readinto(my_blob)
                        raw_data = pd.read_csv(blob.name)
                        dfs.append(raw_data)        
                    data = pd.concat(dfs)
                    return data


                def write_to_sql(cursor: pyodbc.Cursor, data: pd.DataFrame, table_name: str) -> None:
                    """Inserts or updates rows of the input DataFrame    """
                    
                    for index, row in data.iterrows():
                        cursor.execute(f'''MERGE INTO {table_name} as Target
                        USING (SELECT * FROM (VALUES (?, ?, ?, ?)) AS s (transaction_id, store, product, price)) AS Source
                        INSERT (transaction_id, store, product, price) VALUES (Source.transaction_id, Source.store, Source.product, Source.price)
                        WHEN MATCHED THEN
                        UPDATE SET transaction_id=Source.transaction_id, store=Source.store, product=Source.product, price=Source.price;
                        ''',
                                    row['transaction_id'],
                                    row['store'],
                                    row['product'],
                                    row['price'],
                                    )


                def main(mytimer: func.TimerRequest) -> None:
                    """The main function expected by the Azure Functions runtime
                    """
                    data = get_data(CONN_STRING, CONTAINER_NAME)
                    data['transaction_id'] = data['transaction_id'].astype(int)
                    data['store'] = data['store'].astype(str)
                    data['product'] = data['product'].astype(str)
                    data['price'] = data['price'].astype(float)
                    data['product'] = data['product'].apply(lambda x: x.replace('shoo', 'shoe').lower())
                    
                    cnxn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DB_NAME};UID={USERNAME};PWD={PASSWORD}', autocommit=True,timeout=60)
                    cursor = cnxn.cursor()
                    write_to_sql(cursor, data, TABLE_NAME)


# QueueTrigger -----------------------------------------------------------------------------------------------------------------------------------

    # The QueueTrigger makes it incredibly easy to react to new Queues inside of Azure Queue Storage.

    # Content of "__init__.py":
        import logging
        import azure.functions as func

        def main(msg: func.QueueMessage) -> None:
            logging.info('Python queue trigger function processed a queue item: %s',
                        msg.get_body().decode('utf-8'))
            
            filename = msg.get_body().decode('utf-8')
            logging.info(f"queue trigger function: {filename}")

    # Related "function.json" :
        # For a QueueTrigger to work, you provide a path which dictates where the queue messages are located inside your container.

        {
        "scriptFile": "__init__.py",
        "bindings": [
            {
            "name": "msg",
            "type": "queueTrigger",
            "direction": "in",
            "queueName": "js-queue-items",
            "connection": "funappdemo1217_STORAGE"
            }
        ]
        }


# Blob Trigger ---------------------------------------------------------------------------------------------------------------

    # Example 1: 

        # Content of "__init__.py":
                import logging
                import os, csv, json, traceback
                from time import time
                import xxhash

                from bs4 import BeautifulSoup

                import azure.functions as func
                from azure.storage.blob import BlobServiceClient, ContainerClient

                from shared import pre2017, post2017
                from shared.helpers import *

                def main(myblob: func.InputStream):
                    logging.info(f"Python blob trigger function processed blob \n"
                                f"Name: {myblob.name}\n"
                                f"Blob Size: {myblob.length} bytes")

                    # Get case info from file name, which looks like: case-html/15-1367CR-3:hays:12_13_2022:96316e53a9b706e0.html
                    # First strip off case-html/ from beginning and .html from end of blob name
                    stripped_name = myblob.name.strip("case-html/.")
                    # Then split by : as delimiter
                    file_info = stripped_name.split(":")
                    case_num = file_info[0]
                    county = file_info[1]
                    case_date = file_info[2]
                    html_file_hash = file_info[3][:-5]
                    logging.info(f"Retrieved the following metadata: \n"
                                f"Case Date: {case_num}\n"
                                f"County: {county}\n"
                                f"Date Scraped: {case_date}\n"
                                f"HTML File Hash: {html_file_hash}")
                    

                    # call parser
                    START_TIME = time()     
                    logging.info(f"Processing {case_num} - {county} with {odyssey_version} Odyssey parser...")
                    try:
                        case_soup = BeautifulSoup(myblob, "html.parser", from_encoding="UTF-8")

                        # initialize blob container client for sending json files to
                        blob_connection_str = os.getenv("AzureWebJobsStorage")
                        container_name_json = os.getenv("blob_container_name_json")
                        blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(
                            blob_connection_str
                        )
                        container_client = blob_service_client.get_container_client(container_name_json)

                        # Write JSON data
                        blob_name = f"{case_num}:{county}:{case_date}:{html_file_hash}.json"
                        logging.info(f"Sending {blob_name} to {container_name_json} container...")
                        write_string_to_blob(file_contents=json.dumps(case_data), blob_name=blob_name, container_client=container_client, container_name=container_name_json)

                    except Exception:
                        logging.error(traceback.format_exc())

                    RUN_TIME = time() - START_TIME
                    logging.info(f"Parsing took {RUN_TIME} seconds")


                def write_string_to_blob(
                    file_contents: str, blob_name: str, container_client, container_name: str, overwrite: bool = False
                ) -> bool:
                    blob_client = container_client.get_blob_client(blob_name)
                    if blob_client.exists() and not overwrite:
                        logging.info(msg=f"{blob_name} already exists in {container_name}, skipping.")
                        return False
                    blob_client.upload_blob(data=file_contents)
                    return True

        # Related "function.json" :
                {
                "scriptFile": "__init__.py",
                "bindings": [
                    {
                    "name": "myblob",
                    "type": "blobTrigger",
                    "direction": "in",
                    "path": "case-html/{name}",
                    "connection": "AzureWebJobsStorage"
                    }
                ]
                }

    # Example 2

        # Content of "__init__.py"
            import logging
            import os
            import azure.functions as func
            from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
            from azure.storage.blob import BlobClient, BlobLeaseClient


            def main(myblob: func.InputStream):
                logging.info(f"Python blob trigger function processed blob \n"
                            f"Name: {myblob.name}\n"
                            f"Blob Size: {myblob.length} bytes")
                # Get environment variables 
                source_conn_string = os.environ["source_conn_string"]
                dest_conn_string = os.environ["dest_conn_string"]
                sas_token = os.environ["sas_token"]
                # Assign container names
                source_container_name = "samples-workitems"
                dest_container_name = "outcontainer"

                # Get the blob_name from the event
                blob_name = myblob.name.split("/")[-1]

                try:
                    # Create a BlobClient representing the source blob.
                    source_blob = BlobClient.from_connection_string(source_conn_string, source_container_name, blob_name)

                    # Lease the source blob for the copy operation
                    # to prevent another client from modifying it.
                    lease = BlobLeaseClient(source_blob)
                    # lease.acquire()

                    # Get the source blob's properties and display the lease state.
                    source_props = source_blob.get_blob_properties()
                    print("Lease state: " + source_props.lease.state)

                    # Create a BlobClient representing the
                    # destination blob with a unique name.
                    dest_blob = BlobClient.from_connection_string(dest_conn_string, dest_container_name, blob_name + "-Copy")
                    
                    # Start the copy operation.
                    dest_blob.start_copy_from_url(source_blob.url+sas_token)

                    # Get the destination blob's properties to check the copy status.
                    properties = dest_blob.get_blob_properties()
                    copy_props = properties.copy

                    # Display the copy status.
                    print("Copy status: " + copy_props["status"])
                    print("Copy progress: " + copy_props["progress"])
                    print("Completion time: " + str(copy_props["completion_time"]))
                    print("Total bytes: " + str(properties.size))

                    if (source_props.lease.state == "leased"):
                        # Break the lease on the source blob.
                        lease.break_lease()

                        # Update the destination blob's properties to check the lease state.
                        source_props = source_blob.get_blob_properties()
                        print("Lease state: " + source_props.lease.state)

                except ResourceNotFoundError as ex:
                    print("ResourceNotFoundError: ", ex.message)

                except ServiceRequestError as ex:
                    print("ServiceRequestError: ", ex.message)
