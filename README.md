# ROR affiliation matching w/ NER fallback

## Overview
Example scripts for matching affiliations to their corresponding ROR IDs using Named Entity Recognition (NER) model fallback logic. If a direct match is not found using the standard affiliation route/parsing, the model is invoked to extract relevant organization names and locations from the affiliation string and retry the query.

## Installation

```bash
pip install -r requirements.txt
```

## Usage
- `-i`, `--input`: Path to the input CSV file. (Required)
- `-o`, `--output`: Path to the output CSV file. (Default: `affiliation_matching_results.csv`)

### Example
```bash
python ror_affiliation_matching_ner_fallback.py -i input.csv -o output.csv
```

## Input File Format
The input file should have the following columns:
- `affiliation`: The affiliation string to be matched.
- `id`: The expected ROR ID.

## Output File Format
The output file will include the following columns:
- `query_url`: The URL for the ROR API query.
- `match`: Indicates if the affiliation matched the expected ROR ID from the input file (`Y`, `N`, or `NP`).
- `predicted_ror_id`: The ROR ID predicted by the script.
- `error`: Indicates if an error occurred during processing.

## Logging
Errors are logged to a file named with the current timestamp and `_ror_affiliation.log`.

## Test Data
Test data is provided in the `test_data` directory.

## Calculating F-Scores
F-scores can be calculated using the `utilities/calculate_f-score.py` script.

## Notes
Training details for the NER model are available in the [model repo](https://huggingface.co/adambuttrick/ner-test-bert-base-uncased-finetuned-500K-AdamW-3-epoch-locations/blob/main/README.md).
