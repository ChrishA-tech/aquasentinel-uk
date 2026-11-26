"""Evaluate two fixed rules against recorded outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score(frame, loss="absolute"):
    difference=frame[["baseline", "candidate"]].sub(frame["actual"], axis=0)
    errors=difference.abs() if loss=="absolute" else difference.pow(2)
    gap=errors["candidate"]-errors["baseline"]
    if frame.empty or not np.isfinite(errors.to_numpy()).all():
        raise ValueError("No finite scores for the requested benchmark")
    worst=gap.idxmax()
    return {
        "cases":len(frame),
        "baseline_loss":round(float(errors["baseline"].mean()), 6),
        "candidate_loss":round(float(errors["candidate"].mean()), 6),
        "candidate_wins":int((gap < -1e-9).sum()),
        "baseline_wins":int((gap > 1e-9).sum()),
        "ties":int((gap.abs() <= 1e-9).sum()),
        "largest_candidate_disadvantage":round(float(gap.loc[worst]), 6),
        "worst_case":str(worst),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        result=evaluate(args.input)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2,f"Observed benchmark failed: {error}\n")


from zipfile import ZipFile


def evaluate(path):
    with ZipFile(path) as archive:
        with archive.open("United Utilities 2025 Detailed EDM Data.csv") as member:
            data=pd.read_csv(member,encoding="utf-8-sig",usecols=["Unique ID","Discharge Start (GMT)","Discharge Stop (GMT)"])
    start=pd.to_datetime(data["Discharge Start (GMT)"],format="mixed",utc=True,errors="coerce")
    stop=pd.to_datetime(data["Discharge Stop (GMT)"],format="mixed",utc=True,errors="coerce")
    valid=start.notna() & stop.notna() & stop.gt(start) & data["Unique ID"].notna()
    event=pd.DataFrame({"site":data.loc[valid,"Unique ID"],"day":start.loc[valid].dt.floor("D")})
    event=event.loc[event.day.dt.year.eq(2025)].drop_duplicates()
    days=pd.date_range("2025-01-01","2025-12-31",freq="D",tz="UTC")
    daily=event.assign(recorded_start=1).pivot(index="day",columns="site",values="recorded_start").reindex(days).fillna(0)
    # Choose monitors using only the first six months. A zero means no recorded start.
    first_half=daily.loc[:"2025-06-30"]
    eligible=first_half.columns[first_half.sum().ge(10)]
    daily=daily.loc[:,eligible]
    actual=daily.loc["2025-07-01":].stack()
    baseline=daily.shift(1).loc["2025-07-01":].stack()
    candidate=daily.shift(1).rolling(28).mean().loc["2025-07-01":].stack()
    test=pd.DataFrame({"actual":actual,"baseline":baseline,"candidate":candidate})
    test.index=test.index.set_names(["date","site"])
    return {"source":"Environment Agency 2025 detailed EDM start/stop archive, United Utilities",
            "metric":"Brier score for a recorded spill start on a UTC site-day",
            "baseline":"Previous day's binary recorded-start flag","candidate":"Previous 28 days' recorded-start rate",
            "eligible_monitors":len(eligible),"invalid_source_rows":int((~valid).sum()),
            "recorded_start_rate":round(float(test.actual.mean()),6),"holdout":["2025-07-01","2025-12-31"],
            "metrics":score(test,"brier"),
            "by_month":{str(k):score(v,"brier") for k,v in test.groupby(test.index.get_level_values("date").strftime("%Y-%m"))}}


if __name__=="__main__":
    main()
