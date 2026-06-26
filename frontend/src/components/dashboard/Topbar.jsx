import React from "react";
import styles from "./Topbar.module.css";

export default function Topbar({ apiStatus, onRefresh, loading }) {
  const dotColor = apiStatus === "online" ? "#22C55E"
                 : apiStatus === "offline" ? "#EF4444" : "#F59E0B";
  return (
    <header className={styles.topbar}>
      <div className={styles.logo}>
        <div className={styles.icon}>🏇</div>
        <span className={styles.name}>Race<span>Oracle</span></span>
        <span className={styles.badge}>AI</span>
      </div>
      <div className={styles.right}>
        <span className={styles.dot} style={{ background: dotColor, boxShadow: `0 0 6px ${dotColor}` }} />
        <span className={styles.statusText}>API {apiStatus}</span>
        <button className={styles.btn} onClick={onRefresh} disabled={loading}>
          {loading ? "Loading…" : "↺ Refresh"}
        </button>
      </div>
    </header>
  );
}
