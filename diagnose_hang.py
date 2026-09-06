# -*- coding: utf-8 -*-
import time
import sys

def timed_step(name, func):
    print(f"=== START: {name} ===", flush=True)
    t0 = time.time()
    try:
        res = func()
        elapsed = time.time() - t0
        print(f"=== END: {name} (Elapsed: {elapsed:.3f}s) | Result: {res} ===\n", flush=True)
        return res
    except Exception as e:
        elapsed = time.time() - t0
        print(f"=== FAILED: {name} (Elapsed: {elapsed:.3f}s) | Error: {e} ===\n", flush=True)

# 1. import cv2
timed_step("1. import cv2", lambda: __import__("cv2").__version__)

# 2. import torch
timed_step("2. import torch", lambda: __import__("torch").__version__)

# 3. import config
def step3():
    import config
    return f"CHROMA_PERSIST_DIR={config.CHROMA_PERSIST_DIR}"
timed_step("3. import config", step3)

# 4. import sqlite/chroma dependencies
def step4():
    import sqlite3
    import chromadb
    return f"sqlite3={sqlite3.sqlite_version}, chromadb={chromadb.__version__}"
timed_step("4. import sqlite/chroma dependencies", step4)

# 5. open SQLite
def step5():
    from config import SQLITE_DB_PATH
    from storage.sqlite_db import SQLiteDatabase
    db = SQLiteDatabase(SQLITE_DB_PATH)
    with db.connection() as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    return f"Tables count: {len(tables)}, names: {tables[:5]}"
timed_step("5. open SQLite", step5)

# 6. initialize/connect to Chroma
def step6():
    from config import CHROMA_PERSIST_DIR
    from storage.vector_db import VectorDatabase
    vdb = VectorDatabase(CHROMA_PERSIST_DIR)
    return f"Collections: {[c.name for c in vdb.client.list_collections()]}"
timed_step("6. initialize/connect to Chroma", step6)

# 7. inspect collection names/counts
def step7():
    from config import CHROMA_PERSIST_DIR
    from storage.vector_db import VectorDatabase
    vdb = VectorDatabase(CHROMA_PERSIST_DIR)
    counts = {c.name: c.count() for c in vdb.client.list_collections()}
    return f"Collection counts: {counts}"
timed_step("7. inspect collection names/counts", step7)
