Run Hybrid approach solver (ICP+ GA)

```python
cd code
```

```python
python3 solverHybrid.py --instance_file ../instances/ClassicBenchmark/jobshop_swv14 --limit 1 --time_limit 400
```

```python
python3 solverHybrid.py --instance_file ../instances/ClassicBenchmark/jobshop_swv14 --time_limit 400
```

Additional optional arguments
- `--instance_file` : Path to the instance file
- `--time_limit 60` : Time limit in seconds (default: 60)
- `--limit 0` : Set number of solution to use in case of CP-SAT with limiter
- `--output filename` : Base name for output files (default: scheduleHybrid)

If the solution limit is provided, it use ICP solver with solution limit. Otherwise it use ICP solver with time limit

---

Run CP-sat alone solver 

```python
cd code
```

```python
python3 solverICP.py ../instances/test1
```

Run GA alone solver 

```python
cd code
```

```python
python3 solverGA.py ../instances/test1
```