import { useState, useCallback } from "react";
import { api } from "../services/api";

export function useRacePrediction() {
  const [race,    setRace]    = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const loadDemo = useCallback(async () => {
    setLoading(true); setError(null);
    try   { setRace(await api.demo()); }
    catch (e) { setError(e.message); }
    finally   { setLoading(false); }
  }, []);

  const predict = useCallback(async (horses, race) => {
    setLoading(true); setError(null);
    try   { setRace(await api.predict({ horses, race, fetch_news: false })); }
    catch (e) { setError(e.message); }
    finally   { setLoading(false); }
  }, []);

  return { race, loading, error, loadDemo, predict };
}

export function useHealth() {
  const [status, setStatus] = useState("checking");
  const check = useCallback(async () => {
    try   { const h = await api.health(); setStatus(h.status === "ok" ? "online" : "degraded"); }
    catch { setStatus("offline"); }
  }, []);
  return { status, check };
}
