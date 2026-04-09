# test_rgb.py

View one frame from Basler in gray scale

# test_dvs.py

**Explanation of EventsIterator and event instances**

```python
mv_it = EventsIterator(str(paths[0]))

for ev in mv_it:
```

**One instance of ev**:

- ev["x"] → array([120, 305, 89, ...]) # pixel x-coordinates
- ev["y"] → array([200, 150, 340, ...]) # pixel y-coordinates
- ev["t"] → array([6154, 6180, 6205, ...]) # timestamps in microseconds
- ev["p"] → array([1, 0, 1, ...]) # polarity (ON=1, OFF=0)

Each iteration gives you one batch of events (10k-100k events)

**Example**:

```python
# Batch 1: 50k events from t=6154 to t=50200
ev["t"] = [6154, 6180, 6205, ..., 50200]  # len = 50000
ev["x"] = [120, 305, 89, ..., 440]

# Batch 2: 50k events from t=50201 to t=95800
ev["t"] = [50201, 50230, 50255, ..., 95800]  # len = 50000

# Batch 3: 23k events from t=95801 to t=120000
ev["t"] = [95801, 95830, ..., 120000]  # len = 23000
```
