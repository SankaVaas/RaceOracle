import React from "react";
import s from "./Sidebar.module.css";

const RACES = [
  { id:"demo", label:"Royal Ascot R3", meta:"1m4f · Good · 4 runners", badge:"Demo" },
];

export default function Sidebar({ active, onSelect }) {
  return (
    <aside className={s.sidebar}>
      <div className={s.sectionLabel}>Upcoming Races</div>
      {RACES.map(r => (
        <div key={r.id} className={`${s.card} ${active === r.id ? s.active : ""}`} onClick={() => onSelect(r.id)}>
          <div className={s.cardName}>{r.label}</div>
          <div className={s.cardMeta}>{r.meta}</div>
          <span className={s.cardBadge}>{r.badge}</span>
        </div>
      ))}
      <div className={s.hint}>
        POST to <code>/api/v1/predict</code> to add real races.
      </div>
    </aside>
  );
}
