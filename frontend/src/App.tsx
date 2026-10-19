import { useState } from "react";
import { processImage, b64ToDataUrl } from "./api";
import type { ProcessResponse } from "./types";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ProcessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFile(e.target.files?.[0] ?? null);
    setResult(null);
    setError(null);
  }

  async function onProcess() {
    if (!file) {
      setError("выберите файл");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await processImage(file);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1>kifu-to-sgf: тест обработки изображения</h1>

      <div style={{ marginBottom: 16 }}>
        <input
          type="file"
          accept="image/jpeg,image/png"
          onChange={onChange}
        />
        <button
          onClick={onProcess}
          disabled={!file || loading}
          style={{ marginLeft: 8 }}
        >
          {loading ? "Обрабатываем..." : "Обработать"}
        </button>
      </div>

      {error && <p style={{ color: "crimson" }}>Ошибка: {error}</p>}

      {result && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 16,
            marginTop: 16,
          }}
        >
          <figure style={{ margin: 0 }}>
            <figcaption>Оригинал ({result.width}x{result.height})</figcaption>
            <img
              src={b64ToDataUrl(result.original_png_b64)}
              alt="original"
              style={{ maxWidth: "100%", border: "1px solid #ccc" }}
            />
          </figure>
          <figure style={{ margin: 0 }}>
            <figcaption>Обработано (reverse pixels)</figcaption>
            <img
              src={b64ToDataUrl(result.processed_png_b64)}
              alt="processed"
              style={{ maxWidth: "100%", border: "1px solid #ccc" }}
            />
          </figure>
        </div>
      )}
    </main>
  );
}