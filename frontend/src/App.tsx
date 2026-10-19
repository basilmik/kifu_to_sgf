import { useState } from "react";
import { b64ToDataUrl, debugDetect, detectBoard, processImage } from "./api";
import type {
  DebugDetectResponse,
  DetectBoardResponse,
  ProcessResponse,
} from "./types";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [processing, setProcessing] = useState(false);
  const [reverseResult, setReverseResult] = useState<ProcessResponse | null>(null);

  const [detecting, setDetecting] = useState(false);
  const [detectResult, setDetectResult] = useState<DetectBoardResponse | null>(null);

  const [debugging, setDebugging] = useState(false);
  const [debugResult, setDebugResult] = useState<DebugDetectResponse | null>(null);

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFile(e.target.files?.[0] ?? null);
    setReverseResult(null);
    setDetectResult(null);
    setDebugResult(null);
    setError(null);
  }

  async function onReverse() {
    if (!file) return;
    setProcessing(true); setError(null);
    try { setReverseResult(await processImage(file)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setProcessing(false); }
  }

  async function onDetect() {
    if (!file) return;
    setDetecting(true); setError(null);
    try { setDetectResult(await detectBoard(file)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setDetecting(false); }
  }

  async function onDebug() {
    if (!file) return;
    setDebugging(true); setError(null);
    try { setDebugResult(await debugDetect(file)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setDebugging(false); }
  }

  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1>kifu-to-sgf</h1>

      <div style={{ marginBottom: 16 }}>
        <input type="file" accept="image/jpeg,image/png" onChange={onChange} />
        <button onClick={onReverse} disabled={!file || processing} style={{ marginLeft: 12 }}>
          {processing ? "Обрабатываем..." : "Реверс пикселей"}
        </button>
        <button onClick={onDetect} disabled={!file || detecting} style={{ marginLeft: 8 }}>
          {detecting ? "Ищем доску..." : "Найти доску"}
        </button>
        <button onClick={onDebug} disabled={!file || debugging} style={{ marginLeft: 8 }}>
          {debugging ? "Debug..." : "Debug detect"}
        </button>
      </div>

      {error && <p style={{ color: "crimson" }}>Ошибка: {error}</p>}

      {reverseResult && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 12 }}>
          <figure style={{ margin: 0 }}>
            <figcaption>Оригинал ({reverseResult.width}x{reverseResult.height})</figcaption>
            <img src={b64ToDataUrl(reverseResult.original_png_b64)} alt="orig" style={{ maxWidth: "100%" }} />
          </figure>
          <figure style={{ margin: 0 }}>
            <figcaption>Реверс</figcaption>
            <img src={b64ToDataUrl(reverseResult.processed_png_b64)} alt="rev" style={{ maxWidth: "100%" }} />
          </figure>
        </div>
      )}

      {detectResult && (
        <>
          <p>
            {detectResult.found
              ? `Доска найдена. Углы: ${detectResult.corners.map(([x, y]) => `(${x}, ${y})`).join(", ")}`
              : "Доска не найдена."}
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <figure style={{ margin: 0 }}>
              <figcaption>Оригинал</figcaption>
              <img src={b64ToDataUrl(detectResult.original_png_b64)} alt="orig" style={{ maxWidth: "100%" }} />
            </figure>
            <figure style={{ margin: 0 }}>
              <figcaption>Найденная доска</figcaption>
              <img src={b64ToDataUrl(detectResult.marked_png_b64)} alt="marked" style={{ maxWidth: "100%" }} />
            </figure>
          </div>
        </>
      )}

      {debugResult && (
        <section style={{ marginTop: 24, borderTop: "2px solid #ddd", paddingTop: 16 }}>
          <h2>Debug</h2>
          <p><b>note:</b> {debugResult.note}</p>
          <p><b>found:</b> {String(debugResult.found)}</p>
          <p><b>img_area:</b> {debugResult.img_area.toFixed(0)} px²</p>

          <h3>Промежуточные изображения</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
            {debugResult.images.map((img) => (
              <figure key={img.name} style={{ margin: 0 }}>
                <figcaption>{img.name} ({img.width}x{img.height})</figcaption>
                <img src={b64ToDataUrl(img.png_b64)} alt={img.name} style={{ maxWidth: "100%", border: "1px solid #ccc" }} />
              </figure>
            ))}
          </div>

          <h3>Топ контуров по площади</h3>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr>
                <th style={cell}>#</th>
                <th style={cell}>area</th>
                <th style={cell}>area/img_area</th>
                <th style={cell}>perimeter</th>
                <th style={cell}>approx_pts</th>
                <th style={cell}>reasonable</th>
              </tr>
            </thead>
            <tbody>
              {debugResult.top_contours.map((c, i) => (
                <tr key={i}>
                  <td style={cell}>{i}</td>
                  <td style={cell}>{c.area.toFixed(0)}</td>
                  <td style={cell}>{(c.area / debugResult.img_area * 100).toFixed(1)}%</td>
                  <td style={cell}>{c.perimeter.toFixed(1)}</td>
                  <td style={cell}>{c.approx_points}</td>
                  <td style={{ ...cell, color: c.reasonable ? "green" : "crimson" }}>
                    {String(c.reasonable)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </main>
  );
}

const cell: React.CSSProperties = {
  border: "1px solid #ccc",
  padding: "4px 8px",
  textAlign: "left",
};