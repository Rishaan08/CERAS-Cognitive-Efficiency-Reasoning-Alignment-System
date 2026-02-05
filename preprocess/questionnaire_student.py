# Imports
from pathlib import Path
import duckdb
import time

# Paths
data_dir = Path("./data/processed")
qqq_file = data_dir / "pisa_2022_student_qqq.parquet"
output = data_dir / "qqq_student_features.parquet"

def log(msg):
    print(msg)

def build_questionnaire_student_features():

    start = time.time()
    log("Building QQQ student-level features")

    # DuckDB setup
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads=1")
    con.execute("SET memory_limit='8GB'")
    con.execute("SET preserve_insertion_order=false")

    # Load parquet
    log("Loading QQQ parquet...")
    con.execute(f"""
        CREATE TABLE qqq_raw AS
        SELECT * FROM read_parquet('{qqq_file}')
    """)

    # Normalize columns names to lowercase
    log("Normalizing column names to lowercase...")

    cols = con.execute("PRAGMA table_info('qqq_raw')").fetchall()

    rename_expr = ", ".join(
        f'"{c[1]}" AS {c[1].lower()}'
        for c in cols
    )

    con.execute(f"""
        CREATE TABLE qqq AS
        SELECT {rename_expr}
        FROM qqq_raw
    """)

    con.execute("DROP TABLE qqq_raw")

    # Detect student ID
    res = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'qqq'
          AND column_name LIKE '%stuid%'
        LIMIT 1
    """).fetchone()

    if res is None:
        raise ValueError("Student ID column not found")

    student_id_col = res[0]
    log(f"Using student ID column: {student_id_col}")

    # Detect numeric questionnaire columns
    num_cols = con.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'qqq'
          AND column_name != '{student_id_col}'
          AND data_type IN ('INTEGER','BIGINT','DOUBLE','FLOAT','REAL')
    """).fetchall()

    num_cols = [c[0] for c in num_cols]
    log(f"Numeric questionnaire columns: {len(num_cols)}")

    if len(num_cols) < 50:
        log("WARNING: Low numeric feature count, but continuing")

    # SQL-only student-level aggregation
    log("Computing student-level features")

    con.execute(f"""
        COPY (
            WITH exploded AS (
                SELECT
                    {student_id_col},
                    UNNEST([{",".join(num_cols)}]) AS v
                FROM qqq
            ),
            ...
        )
        TO '{output}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output}')"
    ).fetchone()[0]

    print("\nFINAL QQQ STUDENT-LEVEL DATASET")
    print(f"Rows: {rows:,}")
    print("Saved to:", output)
    print(f"Time taken: {time.time() - start:.2f}s")

    con.close()

if __name__ == "__main__":
    build_questionnaire_student_features()