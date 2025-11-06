import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def to_snake_case(s: str) -> str:
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"__+", "_", s)
    return s.strip().lower()

def simple_impute(df: pd.DataFrame, numeric_strategy="median") -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                if numeric_strategy == "median":
                    val = df[col].median()
                elif numeric_strategy == "mean":
                    val = df[col].mean()
                else:
                    val = 0
                df[col] = df[col].fillna(val)
        else:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
            else:
                if df[col].isnull().any():
                    df[col] = df[col].astype(object).fillna("unknown")
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notnull().sum() > 0.8 * len(df):
                df[col] = parsed
        except Exception:
            continue
    return df

def clean_csv(infile: str, outfile: str, numeric_strategy: str = "median") -> None:
    df = pd.read_csv(infile)
    df.columns = [to_snake_case(c) for c in df.columns]
    df = coerce_types(df)
    df = simple_impute(df, numeric_strategy=numeric_strategy)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"Saved cleaned CSV to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw csv path")
    parser.add_argument("--output", required=True, help="clean csv output path")
    parser.add_argument("--numeric-strategy", default="median", choices=["median","mean","zero"])
    args = parser.parse_args()
    clean_csv(args.input, args.output, args.numeric_strategy)
