import React from "react";
import s from "./NewsIntel.module.css";

const SEV_COLOR = { critical:"#EF4444", high:"#F59E0B", medium:"#F59E0B", low:"#8C97B0" };

export default function NewsIntel({ riskFlags = [], newsSummary = "" }) {
  return (
    <div>
      <div className={s.title}>News intelligence</div>
      {riskFlags.map((f, i) => (
        <div className={s.item} key={i}>
          <div className={s.dot} style={{ background: SEV_COLOR[f.severity] || "#8C97B0" }} />
          <span className={s.text}>{f.label}</span>
        </div>
      ))}
      {newsSummary && (
        <div className={s.item}>
          <div className={s.dot} style={{ background:"#4A5568" }} />
          <span className={s.text}>{newsSummary}</span>
        </div>
      )}
      {!riskFlags.length && !newsSummary && (
        <div className={s.empty}>No news flags detected</div>
      )}
    </div>
  );
}
