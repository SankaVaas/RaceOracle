// Use relative path so Vite proxy forwards to FastAPI on port 8000
const BASE = "/api/v1";

async function request(method, path, body = null) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(e.detail || `API error ${res.status}`);
  }
  return res.json();
}

export const api = {
  health:  ()     => request("GET",  "/health"),
  demo:    ()     => request("GET",  "/predict/demo"),
  predict: (body) => request("POST", "/predict", body),
  news:    (body) => request("POST", "/news", body),
};