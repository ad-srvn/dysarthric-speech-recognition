# TORGO data

The TORGO corpus is not included in this repository. Download it under its own
terms and extract the speaker folders here. The expected layout is:

```text
data/
├── F/F01/Session1/...
├── M/M01/Session1/...
├── FC/FC01/Session1/...
└── MC/MC01/Session1/...
```

`F` and `M` contain dysarthric speakers. `FC` and `MC` contain control
speakers. The pipeline searches recursively, so an extra archive directory
between these levels is acceptable.

Do not commit corpus recordings or transcripts. Everything in this directory
except this file and `.gitkeep` is ignored by Git.
