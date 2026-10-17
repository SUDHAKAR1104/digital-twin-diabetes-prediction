import pandas as pd


def load_patient_data(file_path):
    """Load the patient dataset from a CSV file."""
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Rows and columns:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    return df
