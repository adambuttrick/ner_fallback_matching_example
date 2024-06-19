# Calculate F-scores

## Overview
Calculates precision, recall, F1 score, F0.5 score, and specificity for a given CSV file containing affiliation matching results.

## Usage
- `-i`, `--input`: Path to the input CSV file. (Required)
- `-o`, `--output`: Path to the output CSV file. (Default: `<input_file>_metrics.csv`)

### Example
```bash
python calculate_f-score.py -i input.csv -o metrics.csv
```

## Input File Format
The input file should have the following columns:
- `predicted_ror_id`: The ROR ID predicted by the affiliation matching script.
- `id`: The expected ROR ID.

## Output File Format
The output CSV file will include the following metrics:
- Precision
- Recall
- F1 Score
- F0.5 Score
- Specificity
