import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Dataset path")
    parser.add_argument("--output-path", help="Dataset output path")

    args = parser.parse_args()

    data_path = args.input_path
    output_path = args.output_path

    df = pd.read_csv(data_path)

    encoder = LabelEncoder()

    df['Sex'] = encoder.fit_transform(df['Sex'])
    df = df.drop(['Name'], axis=1)
    scalar = StandardScaler()
    
    df = scalar.fit_transform(df)

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess()