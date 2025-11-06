import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def get_pg_engine():
    user = os.environ.get("PGUSER", "postgres")
    pwd  = os.environ.get("PGPASSWORD", "")
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    db   = os.environ.get("PGDATABASE", "mydb")
    return create_engine(f"postgresql://{user}:{pwd}@{host}:{port}/{db}", echo=False)

CREATE_SQL = Path("sql/create_customer_table.sql").read_text()

def create_table_if_not_exists(engine):
    with engine.begin() as conn:
        conn.execute(text(CREATE_SQL))
    print("Ensured customer_churn table exists.")

def upload_csv_to_db(csv_path: str, table_name: str = "customer_churn") -> None:
    engine = get_pg_engine()
    create_table_if_not_exists(engine)
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if 'senior_citizen' in df.columns:
        df['senior_citizen'] = df['senior_citizen'].apply(lambda x: bool(int(x)) if str(x).isdigit() else (True if str(x).lower() in ['true','yes','y','1'] else False if str(x).lower() in ['false','no','n','0'] else None))
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Uploaded {len(df)} rows to {table_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="cleaned csv to upload")
    parser.add_argument("--table", default="customer_churn", help="target table name")
    args = parser.parse_args()
    upload_csv_to_db(args.csv, args.table)
