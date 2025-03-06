Run Hybrid approach solver (ICP+ GA)

```python
cd code
```

```python
python3 solverHybrid.py ../instances/test1
```

Additional optional arguments
- `--time_limit 60` : Time limit in seconds (default: 60)

- `--output filename` : Base name for output files (default: scheduleHybrid)

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