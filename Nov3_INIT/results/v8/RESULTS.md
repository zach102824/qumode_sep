# N3INIT v8 L=2 holdout

Exact v4 four-start recipes. Fresh holdout seeds 10144–10151. Not GR. Not development 10018–10025.
Gates H1/H2 use Q=2048 only. Both q=512 and q=1024 were run.

New quantum queries 98304. Historical 1949696. Total research queries 2048000. Failures 0.
Classical B* wall 966.3s, CPU 965.3s, peak RSS 179656 KiB.

**L2-holdout FAIL.** H1 N15 Q2048 `False`. H2 N18 Q2048 `False`.

| N | q | Q | n | mean (E−E0)/N Nov3 | QAOA | mean F Nov3 | QAOA | mean ΔE | CI | median | min | max | wins | start0 Nov3 | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 15 | 512 | 1024 | 8 | 0.0332 | 0.0360 | 0.686 | 0.665 | -0.0422 | [-0.112, +0.021] | -0.0228 | -0.2049 | +0.0645 | 4/8 | 5/8 | report |
| 15 | 1024 | 2048 | 8 | 0.0311 | 0.0332 | 0.707 | 0.681 | -0.0311 | [-0.083, +0.014] | +0.0028 | -0.1670 | +0.0350 | 4/8 | 4/8 | H1 |
| 18 | 512 | 1024 | 8 | 0.0353 | 0.0374 | 0.658 | 0.639 | -0.0378 | [-0.152, +0.076] | -0.0030 | -0.3497 | +0.1948 | 4/8 | 4/8 | report |
| 18 | 1024 | 2048 | 8 | 0.0332 | 0.0368 | 0.671 | 0.643 | -0.0650 | [-0.184, +0.045] | -0.0361 | -0.3679 | +0.1391 | 5/8 | 6/8 | H2 |

Do not patch PR #77 from this file.
Fit cell time 5431.5s. Workers 4, BLAS 1.
