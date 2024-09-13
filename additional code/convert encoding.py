import pandas as pd
import chardet

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']        
        return encoding, confidence

def convert_encoding(input_file, output_file, encoding, to_encoding='utf-8'):
    # Read the file with the current encoding
    df = pd.read_csv(input_file, encoding=encoding)
    
    # Save the file with the new encoding
    df.to_csv(output_file, encoding=to_encoding, index=False)

# Example usage
input_file = 'output_default.csv'
output_file = 'output_default_utf8.csv'

encoding, confidence = detect_file_encoding(input_file)
print(f"The detected encoding is {encoding} with a confidence of {confidence:.2f}")

convert_encoding(input_file, output_file, encoding)
