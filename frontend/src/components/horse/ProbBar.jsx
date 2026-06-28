import React from "react";
import s from "./ProbBar.jsx";

export default function ProbBar({ label, pct, color }) {
  return (
    <div className={s.row}>
      <span className={s.label}>{label}</span>
      <div className={s.track}>
        <div className={s.fill} style={{ width:`${pct}%`, background:color }} />
      </div>
      <span className={s.pct} style={{ color }}>{pct}%</span>
    </div>
  );
}