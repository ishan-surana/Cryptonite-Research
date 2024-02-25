import bson
import csv

def bson_to_csv(input_file, output_file):
    with open(input_file, 'rb') as f:
        # Load BSON data
        bson_data = f.read()

        # Open CSV file for writing
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Decode BSON data
            while bson_data:
                # Extract BSON document length
                record_length = bson_data[:4]
                document_length = int.from_bytes(record_length, 'little')

                # Decode BSON document
                record = bson.loads(bson_data[:document_length])

                # Write BSON document fields to CSV
                writer.writerow([record.get('_id', ''), record.get('date', ''), record.get('id', ''), record.get('relevant', ''), record.get('text', ''), record.get('tweet', ''), record.get('type', ''), record.get('watson', ''), record.get('annotation', '')])

                # Move to the next BSON document
                bson_data = bson_data[document_length:]

# Convert BSON to CSV
bson_to_csv('tweets.bson', 'tweets.csv')