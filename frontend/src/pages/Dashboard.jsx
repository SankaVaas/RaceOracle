import React, { useEffect, useState } from "react";
import { useRacePrediction, useHealth } from "../hooks/useRacePrediction";
import Topbar       from "../components/dashboard/Topbar";
import Sidebar      from "../components/dashboard/Sidebar";
import MetricsRow   from "../components/dashboard/MetricsRow";
import HorseCard    from "../components/horse/HorseCard";
import BottomPanels from "../components/dashboard/BottomPanels";
import s from "./Dashboard.module.css";

export default function Dashboard() {
  const { race, loading, error, loadDemo } = useRacePrediction();
  const { status, check }                   = useHealth();
  const [expanded, setExpanded]             = useState(null);
  const [activeRace, setActiveRace]         = useState("demo");

  useEffect(() => { check(); loadDemo(); }, []);

  const toggle = i => setExpanded(prev => prev === i ? null : i);

  return (
    <div className={s.app}>
      <Topbar apiStatus={status} onRefresh={loadDemo} loading={loading} />
      <div className={s.body}>
        <Sidebar active={activeRace} onSelect={id => { setActiveRace(id); loadDemo(); }} />
        <main className={s.main}>
          {error && <div className={s.error}>⚠ {error} — is the API running on port 8000?</div>}

          {loading && (
            <div className={s.loading}>
              <div className={s.spinner} />
              <span>Running AI prediction…</span>
            </div>
          )}

          {!loading && race && (
            <>
              <MetricsRow race={race} />

              <div className={s.sectionHdr}>
                <span className={s.sectionTitle}>Field predictions — AI win probability</span>
                <span className={s.sectionHint}>Click any horse to expand</span>
              </div>

              <div className={s.horses}>
                {race.horses.map((h, i) => (
                  <HorseCard key={h.name} horse={h} expanded={expanded === i} onToggle={() => toggle(i)} />
                ))}
              </div>

              <BottomPanels horses={race.horses} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
