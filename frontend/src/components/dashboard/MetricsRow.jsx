import React from "react";
import s from "./MetricsRow.module.css";

function Metric({ label, value, sub, color }) {
  return (
    <div className={s.card}>
      <div className={s.label}>{label}</div>
      <div className={s.value} style={{ color }}>{value}</div>
      {sub && <div className={s.sub}>{sub}</div>}
    </div>
  );
}

export default function MetricsRow({ race }) {
  if (!race) return null;
  return (
    <div className={s.grid}>
      <Metric label="Race"    value={race.race_name}        sub={`${race.track} · ${race.going}`} />
      <Metric label="Top pick" value={race.top_pick}        sub={`${race.horses[0]?.win_prob}% win prob`} color="var(--gold)" />
      <Metric label="Model confidence" value={`${race.model_confidence}%`} sub={race.model_version} color="var(--purple)" />
      <Metric label="Inference" value={`${race.inference_ms}ms`} sub={`${race.field_size} runners`} color="var(--green)" />
    </div>
  );
}
