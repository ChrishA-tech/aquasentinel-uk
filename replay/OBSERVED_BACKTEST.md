# Observed-outcome benchmark

Download the 2025 detailed ZIP, including the United Utilities member. This compares the previous day’s binary start flag with the preceding 28-day start frequency. The July–December 2025 test includes monitors selected using only January–June activity. The target is at least one recorded start on a UTC site-day.

| Scored cases | Baseline mean loss | Candidate mean loss | Candidate wins | Baseline wins | Ties |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 121,624 | 0.237560 | 0.154043 | 28,352 | 83,467 | 9,805 |

**Metric:** Brier score for a recorded spill start on a UTC site-day. Lower is better. See [aggregate results](observed-results.json) for the largest regression and breakdowns.

The rolling frequency reduced average squared probability error even though the binary baseline won on more individual days. A day without a recorded start is coded zero. Monitor downtime and unreported events may produce missing labels. This benchmark does not measure pollution, permit compliance or hourly warning lead time.

## Reproduce

Save [Environment Agency 2025 detailed EDM archive](https://www.data.gov.uk/dataset/39ba1e24-69ca-4031-badf-e02649b74a7b/event-duration-monitoring-storm-overflow-start-stop-detailed-data) as `edm.zip`. The input file SHA-256 is `75b8a7fa25970812a6287c26e010200da4373b55383c98d2344dfa33188bc001`. The workbook requires `xlrd`; all benchmarks use `pandas` and `numpy`.

```sh
python replay/observed_backtest.py edm.zip --output /tmp/observed-results.json
diff -u replay/observed-results.json /tmp/observed-results.json
```
