# pip install pyarrow

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds # partitioning of big data
import pyarrow.compute as pc

days = pa.array([1, 12, 17, 23, 28], type=pa.int8())
months = pa.array([1, 3, 5, 7, 1], type=pa.int8())
years = pa.array([1990, 2000, 1995, 2000, 1995], type=pa.int16())
birthdays_table = pa.table([days, months, years],names=["days", "months", "years"])

pq.write_table(birthdays_table, 'birthdays.parquet')
reloaded_birthdays = pq.read_table('birthdays.parquet')
ds.write_dataset(birthdays_table, 
                "savedir", # creates sub-folder in current folder
                format="parquet", 
                partitioning=ds.partitioning(pa.schema([birthdays_table.schema.field("years")])))
birthdays_dataset = ds.dataset("savedir", format="parquet", partitioning=["years"])
birthdays_dataset.files


a = pa.array([1, 1, 2, 3, 4, 5, 4, 3])
b = pa.array([4, 1, 2, 1, 2, 5, 3, 8])
pc.sum(a)
pc.equal(a, b)

pc.value_counts(birthdays_table["years"])

x, y = pa.scalar(7.8), pa.scalar(9.3)
pc.multiply(x, y)


import gzip
with gzip.open('example.gz', 'wb') as f:
    f.write(b'some data\n' * 3)
stream = pa.input_stream('example.gz') # contents will automatically be decompressed on reading
stream.read()