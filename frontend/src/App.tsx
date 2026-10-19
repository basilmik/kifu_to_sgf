import { useState } from "react";
import { b64ToDataUrl, detectBoard, processImage } from "./api";
import type { DetectBoardResponse, ProcessResponse } from "./types";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [processing, setProcessing] = useState(false);
  const [reverseResult, setReverseResult] = useState<ProcessResponse | null>(null);

  const [detecting, setDetecting] = useState(false);
  const [detectResult, setDetectResult] = useState<DetectBoardResponse | null>(null);

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFile(e.target.files?.[0] ?? null);
    setReverseResult(null);
    setDetectResult(null);
    setError(null);
  }

  async function onReverse() {
    if (!file) return;
    setProcessing(true);
    setError(null);
    try {
      setReverseResult(await processImage(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setProcessing(false);
    }
  }

  async function onDetect() {
    if (!file) return;
    setDetecting(true);
    setError(null);
    try {
      setDetectResult(await detectBoard(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDetecting(false);
    }
  }

  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1>kifu-to-sgf</h1>

      <div style={{ marginBottom: 16 }}>
        <input type="file" accept="image/jpeg,image/png" onChange={onChange} />
        <button
          onClick={onReverse}
          disabled={!file || processing}
          style={{ marginLeft: 12 }}
        >
          {processing ? "Обрабатываем..." : "Реверс пикселей"}
        </button>
        <button
          onClick={onDetect}
          disabled={!file || detecting}
          style={{ marginLeft: 8 }}
        >
          {detecting ? "Ищем доску..." : "Найти доску"}
        </button>
      </div>

      {error && <p style={{ color: "crimson" }}>Ошибка: {error}</p>}

      {reverseResult && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 16,
            marginTop: 12,
          }}
        >
          <figure style={{ margin: 0 }}>
            <figcaption>
              Оригинал ({reverseResult.width}x{reverseResult.height})
            </figcaption>
            <img
              src={b64ToDataUrl(reverseResult.original_png_b64)}
              alt="original"
              style={{ maxWidth: "100%", border: "1px solid #ccc" }}
            />
          </figure>
          <figure style={{ margin: 0 }}>
            <figcaption>Реверс пикселей</figcaption>
            <img
              src={b64ToDataUrl(reverseResult.processed_png_b64)}
              alt="processed"
              style={{ maxWidth: "100%", border: "1px solid #ccc" }}
            />
          </figure>
        </div>
      )}

      {detectResult && (
        <>
          {detectResult.found ? (
            <p>
              Доска найдена. Углы:{" "}
              {detectResult.corners
                .map(([x, y]) => `(${x}, ${y})`)
                .join(", ")}
            </p>
          ) : (
            <p style={{ color: "darkorange" }}>
              Доска не найдена. Попробуйте другое фото или ракурс.
            </p>
          )}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 16,
              marginTop: 12,
            }}
          >
            <figure style={{ margin: 0 }}>
              <figcaption>
                Оригинал ({detectResult.width}x{detectResult.height})
              </figcaption>
              <img
                src={b64ToDataUrl(detectResult.original_png_b64)}
                alt="original"
                style={{ maxWidth: "100%", border: "1px solid #ccc" }}
              />
            </figure>
            <figure style={{ margin: 0 }}>
              <figcaption>Найденная доска</figcaption>
              <img
                src={b64ToDataUrl(detectResult.marked_png_b64)}
                alt="detected"
                style={{ maxWidth: "100%", border: "1px solid #ccc" }}
              />
            </figure>
          </div>
        </>
      )}
    </main>
  );
}