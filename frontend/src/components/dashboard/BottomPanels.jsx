import React from "react";
import s from "./BottomPanels.module.css";

const ACCURACY = [
  ["Top-pick accuracy",  "38.6%", "var(--green)"],
  ["Top-2 accuracy",     "~62%",  "var(--green)"],
  ["vs random baseline", "3.8×",  "var(--purple)"],
  ["Training races",     "2,706", "var(--text)"],
  ["Dataset period",     "Sep–Dec 2020", "var(--text)"],
];

export default function BottomPanels({ horses = [] }) {
  const maxWin = horses.length ? Math.max(...horses.map(h => h.win_prob)) : 1;
  return (
    <div className={s.grid}>
      <div className={s.panel}>
        <div className={s.title}>Model accuracy tracker</div>
        {ACCURACY.map(([label, val, color]) => (
          <div className={s.row} key={label}>
            <span>{label}</span>
            <span className={s.val} style={{ color }}>{val}</span>
          </div>
        ))}
      </div>
      <div className={s.panel}>
        <div className={s.title}>Field probability overview</div>
        {horses.map(h => (
          <div className={s.fieldRow} key={h.name}>
            <span className={s.fieldName}>{h.name}</span>
            <div className={s.track}>
              <div className={s.fill} style={{ width:`${(h.win_prob/maxWin)*100}%` }} />
            </div>
            <span className={s.pct}>{h.win_prob}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
