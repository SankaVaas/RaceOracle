import React from "react";
import s from "./HorseCard.module.css";
import ProbBar from "./ProbBar";
import ShapChart from "../charts/ShapChart";
import NewsIntel from "../news/NewsIntel";

export default function HorseCard({ horse, expanded, onToggle }) {
  const isTop = horse.rank === 1;
  const winColor = horse.win_prob >= 28 ? "var(--green)"
                 : horse.win_prob >= 18 ? "var(--amber)" : "var(--text2)";

  return (
    <div className={`${s.card} ${expanded ? s.expanded : ""} ${isTop ? s.top : ""}`} onClick={onToggle}>
      {isTop && <div className={s.topLabel}>⭐ AI TOP PICK</div>}

      <div className={s.header}>
        <div className={`${s.num} ${isTop ? s.numGold : ""}`}>{horse.rank}</div>
        <div>
          <div className={s.name}>{horse.name}</div>
          <div className={s.meta}>{horse.jockey} · {horse.trainer} · Form: {horse.form}</div>
        </div>
        <div className={s.badges}>
          {isTop && <span className={`${s.badge} ${s.gold}`}>Top pick</span>}
          {horse.risk_flags?.map((f, i) => (
            <span key={i} className={`${s.badge} ${f.severity === "critical" ? s.red : s.amber}`}>
              {f.label}
            </span>
          ))}
          <span className={`${s.badge} ${s.gray}`}>Conf {horse.confidence}%</span>
        </div>
      </div>

      <ProbBar label="Win"   pct={horse.win_prob}  color={winColor} />
      <ProbBar label="Top 2" pct={horse.top2_prob} color="var(--purple)" />
      <ProbBar label="Top 3" pct={horse.top3_prob} color="var(--text3)" />

      {expanded && (
        <div className={s.details}>
          <ShapChart drivers={horse.top_drivers} risks={horse.top_risks} />
          <div>
            <div className={s.modalTitle}>Modal weights — signal trust</div>
            <div className={s.modals}>
              {["structured","odds","news"].map(k => (
                <div className={s.modalBlock} key={k}>
                  <div className={s.modalVal}>{horse.modal_weights?.[k] ?? 33}%</div>
                  <div className={s.modalLabel}>{k}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop:14 }}>
              <NewsIntel riskFlags={horse.risk_flags} newsSummary={horse.news_summary} />
            </div>
          </div>
        </div>
      )}

      {!expanded && (
        <div className={s.hint}>Click to expand SHAP · news intelligence · modal weights</div>
      )}
    </div>
  );
}
