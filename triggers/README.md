# triggers.py

Extracts the synchronization mapping between your RGB camera and DVS camera by reading hardware trigger timestamps from the `.raw` file.

Keep p = 1 and discard p = 0

The Basler camera sends an electrical pulse to the DVS camera every time it captures a frame. These pulses are recorded in the DVS .raw file as trigger events.

```r
Basler captures frame 0 → sends pulse → DVS records trigger at 1843788 µs
Basler captures frame 1 → sends pulse → DVS records trigger at 1850188 µs
Basler captures frame 2 → sends pulse → DVS records trigger at 1856589 µs
...
Basler captures frame 661 → sends pulse → DVS records trigger at 6074017 µs
```

## Confirmation

Output Example: `rock/r_1`

```r
total unique rising edges: 662
first trigger: 1843788 µs (1.843788s)
last trigger:  6074017 µs (6.074017s)
duration: 4.230229s
First 5 triggers:
  Frame 0: 1843788 µs (1.843788s)
  Frame 1: 1850188 µs (1.850188s)
  Frame 2: 1856589 µs (1.856589s)
  Frame 3: 1862989 µs (1.862989s)
  Frame 4: 1869390 µs (1.869390s)
Last 5 triggers:
  Frame 657: 6048891 µs (6.048891s)
  Frame 658: 6055292 µs (6.055292s)
  Frame 659: 6061692 µs (6.061692s)
  Frame 660: 6068093 µs (6.068093s)
  Frame 661: 6074017 µs (6.074017s)
```

Confirmation:

- 662 RGB frames captured
- 4.23 second recording (6.074017s - 1.843788s = 4.230229s)
- ~6.4ms between frames (1.850188s - 1.843788s = 0.0064 and 1/0.0064 = 156.25)

What this means:

- Before 1843788 µs: DVS is recording, but RGB hasn't started yet → no sync
- 1843788 → 6074017 µs: Both cameras recording, fully synchronized → use this window
- After 6074017 µs: RGB stopped, DVS still recording → no sync
