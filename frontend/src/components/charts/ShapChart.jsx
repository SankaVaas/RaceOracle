import React from "react";
import s from "./ShapChart.module.css";

export default function ShapChart({ drivers = [], risks = [] }) {
  const all = [...drivers, ...risks];
  const maxAbs = Math.max(...all.map(d => Math.abs(d.impact)), 1);

  return (
    <div>
      <div className={s.title}>SHAP — why this probability</div>
      {drivers.slice(0, 4).map((d, i) => (
        <div className={s.row} key={i}>
          <span className={s.label}>{d.feature.replace(/_/g," ")}</span>
          <div className={s.track}>
            <div className={s.fill} style={{ width:`${(Math.abs(d.impact)/maxAbs)*100}%`, background:"#22C55E" }} />
          </div>
          <span className={s.val} style={{ color:"#22C55E" }}>+{d.impact.toFixed(1)}</span>
        </div>
      ))}
      {risks.slice(0, 3).map((r, i) => (
        <div className={s.row} key={i}>
          <span className={s.label}>{r.feature.replace(/_/g," ")}</span>
          <div className={s.track}>
            <div className={s.fill} style={{ width:`${(Math.abs(r.impact)/maxAbs)*100}%`, background:"#EF4444" }} />
          </div>
          <span className={s.val} style={{ color:"#EF4444" }}>{r.impact.toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}
