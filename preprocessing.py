import os
import pandas as pd
import re

# ==========================
# Path Setup
# ==========================
ROOT = os.path.dirname(os.path.abspath(__file__))

# Input files
LIST_FILE = os.path.join(ROOT, 'dataset', 'cxr-study.csv')         # Contains report file paths
RECORD_FILE = os.path.join(ROOT, 'dataset', 'cxr-record.csv')      # Contains image (dicom) paths
SPLIT_FILE = os.path.join(ROOT, 'dataset', 'mimic-split.csv')      # Predefined train/test/validate split
REPORTS_DIR = os.path.join(ROOT, 'dataset', 'mimic-reports', 'files')  # Directory containing actual text reports

# Output files
TRAIN_FILE = os.path.join(ROOT, 'dataset', 'train.csv')
TEST_FILE = os.path.join(ROOT, 'dataset', 'test.csv')
VALIDATION_FILE = os.path.join(ROOT, 'dataset', 'validation.csv')


# ==========================
# Helper Functions
# ==========================

def remove_notification_section(text):
    """
    Removes lines typically found at the end of reports that indicate notification status.
    These aren't useful for findings and may bias the model.
    """
    notification_keywords = [
        "NOTIFICATION", "telephone notification", "Telephone notification",
        "These findings were", "Findings discussed", "Findings were",
        "This preliminary report", "Reviewed with", "A preliminary read"
    ]
    
    for keyword in notification_keywords:
        idx = text.rfind(keyword)
        if idx > 0:
            return text[:idx]
    
    return text


def sanitize(text):
    """
    Cleans and separates a radiology report into "indication" and "findings".
    The split is based on where the word "FINDINGS" appears in the text.

    Returns:
        (indication, findings) if found, else (None, None)
    """
    text = text.strip()
    text = re.sub("\n", " ", text)
    text = re.sub(",", "", text)  # Remove commas to avoid CSV formatting issues

    regex = r'(\bfindings?.?:)'  # Match "FINDINGS:" (case-insensitive)
    match = re.search(regex, text, flags=re.IGNORECASE)

    if match:
        idx = match.start()
        indication = text[:idx].strip()
        findings = text[idx + len(match.group()):].strip()
        findings = remove_notification_section(findings)
        return indication, findings
    else:
        return None, None


def parse_summary(text):
    """
    Further extracts "impression" from the findings section using "IMPRESSION:" or similar.

    Returns:
        [findings, impression] or None if no impression found
    """
    regex = r'impression.?(?::|" ")'  # Match "IMPRESSION:" or "IMPRESSION "
    
    if not re.search(regex, text, flags=re.IGNORECASE):
        return None

    data = re.split(regex, text, flags=re.IGNORECASE)
    data = [d.strip() for d in data if d.strip()]  # Remove empty strings

    if len(data) < 2:
        return None

    findings = data[0]
    impression = " ".join(data[1:])  # Merge multiple segments after impression
    return [findings, impression]


def write_csv(filename, reports):
    """
    Writes the cleaned and split report data into a CSV with headers:
    subject_id, study_id, indication, findings, impression, image_path
    """
    print(f"Writing {filename}...")

    with open(filename, 'w') as f:
        f.write(f"\"subject_id\",\"study_id\",\"indication\",\"findings\",\"impression\",\"image_path\"\n")

        omitted = 0
        progress = 1

        for _, report in reports.iterrows():
            try:
                # Read full text report
                with open(os.path.join(REPORTS_DIR, report['path'])) as x:
                    text = x.read()
            except Exception as e:
                print(f"Could not read report file: {report['path']}. Error: {e}")
                omitted += 1
                continue

            # Split into indication and findings
            indication, text = sanitize(text)
            if text is None:
                omitted += 1
                continue

            if progress % 10000 == 0:
                print(f'Read {progress} files so far...')
            progress += 1

            # Further split into findings and impression
            parsed_data = parse_summary(text)
            if parsed_data is None:
                omitted += 1
                continue

            findings, impression = parsed_data

            # Write to CSV
            f.write(f"\"{report['subject_id']}\",\"{report['study_id']}\",\"{indication}\",\"{findings}\",\"{impression}\",\"{report['image_path']}\"\n")

    print(f"Omitted {omitted} files out of {progress + omitted} total files.")
    print("Done.\n")


def split(split_file, images, reports):
    """
    Joins the images and reports with the predefined MIMIC split and returns
    train, test, validation DataFrames.
    """
    # Join reports and images with split info
    merged_reports = pd.merge(reports, split_file, on=['subject_id', 'study_id'], how='left')
    merged_images = pd.merge(images, split_file, on=['subject_id', 'study_id', 'dicom_id'], how='left')

    # Combine into one table
    combined_df = pd.concat([
        merged_reports,
        merged_images[['path']].rename(columns={'path': 'image_path'})
    ], axis=1)

    # Return split subsets
    return (
        combined_df[combined_df['split'] == 'train'],
        combined_df[combined_df['split'] == 'test'],
        combined_df[combined_df['split'] == 'validate']
    )

# ==========================
# Main Preprocessing Function
# ==========================
def main():
    print("================ Starting data preprocessing ==================")

    print(f"Reading {os.path.basename(LIST_FILE)}, {os.path.basename(RECORD_FILE)}, {os.path.basename(SPLIT_FILE)}...")
    
    # Load raw metadata files
    radiology_reports = pd.read_csv(LIST_FILE)
    images = pd.read_csv(RECORD_FILE)
    split_file = pd.read_csv(SPLIT_FILE)

    # Convert .dcm image paths to .jpg
    images['path'] = images['path'].str.replace('.dcm', '.jpg')

    # Create train/val/test splits
    train, test, validation = split(split_file, images, radiology_reports)

    # Process and save as CSVs
    write_csv(TRAIN_FILE, train)
    write_csv(TEST_FILE, test)
    write_csv(VALIDATION_FILE, validation)

    print("Done.")
    print("==================== End data preprocessing ======================")


# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    main()
