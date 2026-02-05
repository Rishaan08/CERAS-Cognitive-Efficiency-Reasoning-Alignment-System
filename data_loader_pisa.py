#Imports
from pathlib import Path
import duckdb
import time
import threading

#Paths
DATA_DIR = Path("./data/processed")
COG_FILE = DATA_DIR / "pisa_2022_student_cog.parquet"
QQQ_FILE = DATA_DIR / "pisa_2022_student_qqq.parquet"
OUTPUT_FILE = DATA_DIR / "pisa_student.parquet"

#Logger
def log(msg: str):
    print(f"[INFO] {msg}")

#Heartbeat (progress indicator)
def heartbeat(stop_event):
    while not stop_event.is_set():
        print("[INFO] Merge running... still processing")
        time.sleep(30)

#Main Loader
def load_and_merge_pisa():
    start = time.time()
    log("PISA 2022 data loading pipeline (DuckDB | OOM-safe)")

    con = duckdb.connect(database=":memory:")
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='8GB'")

    #Required for large merges
    con.execute("SET temp_directory='./data/tmp'")
    con.execute("PRAGMA max_temp_directory_size='50GB'")

    #Progress & profiling
    con.execute("SET enable_progress_bar=true")
    con.execute("SET progress_bar_time=1000")
    con.execute("SET enable_profiling='json'")
    con.execute("SET profiling_output='duckdb_profile.json'")

    #Load tables
    log("Loading COG parquet...")
    con.execute(f"""
        CREATE TABLE cog AS
        SELECT * FROM read_parquet('{COG_FILE}')
    """)

    log("Loading QQQ parquet...")
    con.execute(f"""
        CREATE TABLE qqq AS
        SELECT * FROM read_parquet('{QQQ_FILE}')
    """)

    #Normalize columns
    for tbl in ["cog", "qqq"]:
        cols = con.execute(f"PRAGMA table_info('{tbl}')").fetchall()
        for _, c, *_ in cols:
            new = c.strip().lower()
            if c != new:
                con.execute(f'ALTER TABLE {tbl} RENAME COLUMN "{c}" TO "{new}"')

        if "ver_dat" in {r[1] for r in cols}:
            con.execute(f"ALTER TABLE {tbl} DROP COLUMN ver_dat")
            log(f"Dropped 'ver_dat' from {tbl.upper()}")

    log("Detecting student-level identifier...")
    student_id = con.execute("""
        SELECT c.column_name
        FROM information_schema.columns c
        JOIN information_schema.columns q
        ON c.column_name = q.column_name
        WHERE c.table_name='cog'
          AND q.table_name='qqq'
          AND c.column_name LIKE '%stuid%'
        LIMIT 1
    """).fetchone()[0]

    log(f"Using student ID column: '{student_id}'")

    # Reduce QQQ width (CRITICAL)
    log("Selecting essential QQQ columns only...")
    qqq_cols = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name='qqq'
          AND (
               column_name LIKE 'escs%'
            OR column_name LIKE 'gender%'
            OR column_name LIKE 'immig%'
            OR column_name LIKE 'fam%'
            OR column_name LIKE 'home%'
            OR column_name LIKE 'ict%'
          )
    """).fetchall()

    qqq_keep = [student_id] + [c[0] for c in qqq_cols]
    qqq_cols_sql = ", ".join(f"q.{c}" for c in qqq_keep if c != student_id)

    log(f"QQQ columns retained: {len(qqq_keep)}")

    #Streamed Merge
    log("Performing memory-safe merge...")

    stop_event = threading.Event()
    hb = threading.Thread(target=heartbeat, args=(stop_event,))
    hb.start()

    con.execute(f"""
        COPY (
            SELECT
                c.*,
                {qqq_cols_sql}
            FROM cog c
            LEFT JOIN qqq q
              ON CAST(c.{student_id} AS VARCHAR)
               = CAST(q.{student_id} AS VARCHAR)
        )
        TO '{OUTPUT_FILE}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    stop_event.set()
    hb.join()

    #Final stats
    con.execute(f"CREATE TABLE merged AS SELECT * FROM read_parquet('{OUTPUT_FILE}')")
    rows = con.execute("SELECT COUNT(*) FROM merged").fetchone()[0]
    cols = len(con.execute("PRAGMA table_info('merged')").fetchall())

    print("\nFinal Dataset Summary")
    print(f"Rows: {rows:,}")
    print(f"Columns: {cols:,}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Time: {time.time() - start:.2f}s")
    print("DuckDB profiling saved to duckdb_profile.json")

    con.close()

#Entry Point
if __name__ == "__main__":
    load_and_merge_pisa()