import { useState } from "react";
import { sum } from "./api";

export default function App() {
  const [a, setA] = useState(2);
  const [b, setB] = useState(3);
  const [result, setResult] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onClick() {
    setLoading(true);
    setError(null);
    try {
      const data = await sum({ a, b });
      setResult(data.result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1>Сложение через core</h1>

      <input
        type="number"
        value={a}
        onChange={(e) => setA(Number(e.target.value))}
      />
      {" + "}
      <input
        type="number"
        value={b}
        onChange={(e) => setB(Number(e.target.value))}
      />
      {" "}
      <button onClick={onClick} disabled={loading}>
        {loading ? "Считаем…" : "Посчитать"}
      </button>

      <p>
        Результат: <b>{result ?? "—"}</b>
      </p>
      {error && <p style={{ color: "crimson" }}>Ошибка: {error}</p>}
    </main>
  );
}