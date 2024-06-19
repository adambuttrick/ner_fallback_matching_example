import csv
import sys
import time
import argparse
import logging
import requests
from datetime import datetime
from urllib.parse import quote
from ner_inference import ner_inference, load_model, load_tokenizer


now = datetime.now()
script_start = now.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f'{script_start}_ror_affiliation.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')


def query_affiliation(affiliation, ror_id, model, tokenizer):
    result = {"url": None, "chosen_id": None, 'error': False}
    try:
        params = {'affiliation': affiliation}
        base_url = "https://api.ror.org/v2/organizations"
        response = requests.get(base_url, params=params)
        result['url'] = response.url
        found_chosen = False
        data = response.json()
        for i, item in enumerate(data["items"]):
            if item['chosen']:
                found_chosen = True
                result.update(
                    {"chosen_id": item["organization"]["id"]})
                break
        # If no ROR ID is returned, use NER model to extract the org name
        # and location from the affiliation string and retry request
        if not found_chosen:
            affiliation = ner_inference(affiliation, model, tokenizer)
            result["ner_inference"] = affiliation
            if affiliation:
                params = {'affiliation': affiliation}
                response = requests.get(base_url, params=params)
                result['url'] = response.url
                data = response.json()
                for i, item in enumerate(data["items"]):
                    if item["organization"]["id"] == ror_id:
                        result.update(
                            {"chosen_id": ror_id if item["chosen"] else None})
                        break
                    elif item['chosen']:
                        result.update(
                            {"chosen_id": item["organization"]["id"]})
                        break
        return result

    except Exception as e:
        logging.error(f'Error for query: {affiliation} - {e}')
        result['error'] = True
        return result


def parse_and_query(input_file, output_file):
    try:
        model = load_model()
        tokenizer = load_tokenizer()
        with open(input_file, 'r+', encoding='utf-8-sig') as f_in, open(output_file, 'w') as f_out:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames + \
                ['query_url', 'match', 'predicted_ror_id', 'error']
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                result = query_affiliation(
                    row['affiliation'], row['id'], model, tokenizer)
                match = "Y" if result['chosen_id'] == row['id'] else "N" if result['chosen_id'] else "NP"
                row.update({
                    'query_url': result['url'],
                    'match': match,
                    'predicted_ror_id': result['chosen_id'],
                    'error': result['error']
                })
                writer.writerow(row)
    except Exception as e:
        logging.error(f'Error in parse_and_query: {affiliation} - {e}')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Return ROR affiliation matches for a given CSV file, with NER fallback for string parsing.')
    parser.add_argument(
        '-i', '--input', help='Input CSV file', required=True)
    parser.add_argument(
        '-o', '--output', help='Output CSV file', default='affiliation_matching_results.csv')
    return parser.parse_args()


def main():
    args = parse_arguments()
    parse_and_query(args.input, args.output)


if __name__ == '__main__':
    main()
