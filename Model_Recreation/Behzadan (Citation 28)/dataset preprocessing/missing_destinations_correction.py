# Data was missing between these rows due to type error and multiple urls
import pandas as pd

# Load data from CSV file
column_names = ['_id', 'date', 'id', 'relevant', 'text', 'tweet', 'type', 'watson', 'annotation']
data = pd.read_csv('tweets_modified_again.csv', names=column_names)

urls=[]
for i in range(12866, 17137):
     urls.append(data.loc[i, 'urls'][2:-2])

final_urls=[]
for url in urls:
	if len(url)>1:
		final_urls.append(url[0])
	else:
		final_urls.append(url)

a = []
for i in range(12866):
    a.append(data.loc[i, 'destination_url'])

final_values = []
for url in final_urls:
    if isinstance(url, str):
        final_values.append(url)
    elif isinstance(url, list):
        if url:  # Check if the list is not empty
            final_values.append(url[0])
        else:
            final_values.append(None)  # Append None for empty lists
    else:
        final_values.append(None)  # Append None for other non-list types

b = a + final_values

# Append destination_url values from `b` to the dataframe `data`
for i, url in enumerate(b):
    data.loc[i, 'destination_url'] = url

# If you want to append the destination_url values from a specific index to the end:
# Specify the starting index based on the length of `data`
start_index = len(data)
for i, url in enumerate(b[start_index:]):
    data.loc[start_index + i, 'destination_url'] = url

data.to_csv('tweets_final.csv', index=False)