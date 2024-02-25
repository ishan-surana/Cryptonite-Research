import bson

def check_field_names(bson_file):
    unique_field_names = set()  # Set to store unique field names
    with open(bson_file, 'rb') as f:
        bson_data = f.read()

        # Decode BSON data
        while bson_data:
            # Extract BSON document length
            record_length = bson_data[:4]
            document_length = int.from_bytes(record_length, 'little')

            # Decode BSON document
            record = bson.loads(bson_data[:document_length])

            # Print field names if they haven't been printed before
            for field_name in record.keys():
                if field_name not in unique_field_names:
                    print(field_name)
                    unique_field_names.add(field_name)

            # Move to the next BSON document
            bson_data = bson_data[document_length:]

            # Check if there is any more data left
            if len(bson_data) <= 4:
                break

# Provide the path to your BSON file
bson_file = 'tweets.bson'
check_field_names(bson_file)
