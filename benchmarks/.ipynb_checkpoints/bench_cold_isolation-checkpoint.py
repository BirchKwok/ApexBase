"""Isolate cold-start SELECT * LIMIT 100 overhead: Rust layer vs Python client layer."""
import gc
import time
import tempfile
import os
import shutil
from apexbase import ApexClient

tmpdir = tempfile.mkdtemp()
db_dir = os.path.join(tmpdir, "db")

# Setup
c = ApexClient(db_dir, drop_if_exists=True)
c.create_table("default")
ROWS = 1_000_000
data = {
    "name": [f"u{i}" for i in range(ROWS)],
    "age": list(range(ROWS)),
    "score": [float(i) for i in range(ROWS)],
    "city": ["Beijing"] * ROWS,
    "category": ["Books"] * ROWS,
}
c.store(data)
c.close()

N = 20

# ── Match main benchmark conditions exactly (gc.collect before timing) ────────
times_match = []
client = None
for _ in range(N):
    # Replicate cold_start_setup
    if client:
        client.close()
    client = ApexClient(db_dir)
    client.use_table("default")
    gc.collect()   # main benchmark does gc.collect() before timer
    gc.collect()   # timer() also does gc.collect() at its start
    t0 = time.perf_counter()
    client.execute("SELECT * FROM default LIMIT 100")
    t1 = time.perf_counter()
    times_match.append((t1 - t0) * 1000)
if client:
    client.close()

# ── Without gc.collect ────────────────────────────────────────────────────────
times_nogc = []
client = None
for _ in range(N):
    if client:
        client.close()
    client = ApexClient(db_dir)
    client.use_table("default")
    t0 = time.perf_counter()
    client.execute("SELECT * FROM default LIMIT 100")
    t1 = time.perf_counter()
    times_nogc.append((t1 - t0) * 1000)
if client:
    client.close()

# ── Rust storage directly (no Python client wrapper overhead) ─────────────────
times_rust = []
client = None
for _ in range(N):
    if client:
        client.close()
    client = ApexClient(db_dir)
    client.use_table("default")
    storage = client._storage
    gc.collect()
    gc.collect()
    t0 = time.perf_counter()
    storage.execute("SELECT * FROM default LIMIT 100")
    t1 = time.perf_counter()
    times_rust.append((t1 - t0) * 1000)
if client:
    client.close()

# ── ApexClient setup time ─────────────────────────────────────────────────────
times_setup = []
prev = None
for _ in range(N):
    if prev:
        prev.close()
    t0 = time.perf_counter()
    c2 = ApexClient(db_dir)
    c2.use_table("default")
    t1 = time.perf_counter()
    times_setup.append((t1 - t0) * 1000)
    prev = c2
if prev:
    prev.close()

s = sorted(times_match)
print(f"Full client (with gc)  min={min(times_match):.3f}  avg={sum(times_match)/N:.3f}  p50={s[N//2]:.3f}  p90={s[int(N*0.9)]:.3f}ms")
s2 = sorted(times_nogc)
print(f"Full client (no gc)    min={min(times_nogc):.3f}  avg={sum(times_nogc)/N:.3f}  p50={s2[N//2]:.3f}  p90={s2[int(N*0.9)]:.3f}ms")
s3 = sorted(times_rust)
print(f"Rust storage (with gc) min={min(times_rust):.3f}  avg={sum(times_rust)/N:.3f}  p50={s3[N//2]:.3f}  p90={s3[int(N*0.9)]:.3f}ms")
s4 = sorted(times_setup)
print(f"ApexClient setup only  min={min(times_setup):.3f}  avg={sum(times_setup)/N:.3f}ms")
print(f"Python wrapper delta   avg={(sum(times_match)-sum(times_rust))/N:.3f}ms")

shutil.rmtree(tmpdir)
