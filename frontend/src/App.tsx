import { useEffect, useState, } from "react";
import { b64ToDataUrl, debugDetect, 
  detectBoard, processImage, detectGrid, recognizeGrid} from "./api";
import type {
  DebugDetectResponse,
  DetectBoardResponse,
  ProcessResponse,
  GridResponse,
  RecognitionResponse,
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

  const [gridProcessing, setGridProcessing] = useState(false);
  const [gridResult, setGridResult] = useState<GridResponse | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [ocrProcessing, setOcrProcessing] = useState(false);

  const [ocrResult, setOcrResult] =
    useState<RecognitionResponse | null>(null);

  useEffect(() => {
  if (!file) {
    setPreviewUrl(null);
    return;
  }

  const url = URL.createObjectURL(file);
  setPreviewUrl(url);

  return () => URL.revokeObjectURL(url);
}, [file]);

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFile(e.target.files?.[0] ?? null);
    setReverseResult(null);
    setDetectResult(null);
    setDebugResult(null);
    setError(null);
    setGridResult(null);
    setOcrResult(null);
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

  async function onGrid() {
    if (!file) return;

    setGridProcessing(true);
    setGridResult(null);
    setError(null);
    setOcrResult(null);
    try {
      const result = await detectGrid(file);
      setGridResult(result);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setGridProcessing(false);
    }
  }

  async function onRecognize() {
    if (!gridResult?.found) return;

    setOcrProcessing(true);
    setOcrResult(null);
    setError(null);

    try {
      const result = await recognizeGrid(gridResult);
      setOcrResult(result);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setOcrProcessing(false);
    }
  }

  function downloadSgf(sgf: string) {
    const blob = new Blob(
      [sgf],
      {
        type: "application/x-go-sgf;charset=utf-8",
      }
    );

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.href = url;
    link.download = "recognized-game.sgf";

    document.body.appendChild(link);
    link.click();
    link.remove();

    window.setTimeout(() => {
      URL.revokeObjectURL(url);
    }, 1000);
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
      <section
        style={{
          marginTop: 24,
          borderTop: "2px solid #ddd",
          paddingTop: 16,
        }}
      >
        <h2>Автоматическое определение сетки</h2>

        <button
          onClick={onGrid}
          disabled={!file || gridProcessing}
        >
          {gridProcessing
            ? "Ищем сетку..."
            : "Найти сетку 19×19"}
        </button>

        {gridResult && (
          <div style={{ marginTop: 20 }}>
            <h3>Результат</h3>

            <p>
              <b>Found:</b> {String(gridResult.found)}
              <br />
              <b>Note:</b> {gridResult.note}
              <br />
              <b>X quality:</b> {gridResult.x_quality.toFixed(3)}
              <br />
              <b>Y quality:</b> {gridResult.y_quality.toFixed(3)}
              <br />
              <b>Пересечений:</b>{" "}
              {gridResult.grid_points_original.length}
            </p>

            {!gridResult.found && (
              <p style={{ color: "darkorange" }}>
                Не удалось подтвердить сетку автоматически.
              </p>
            )}

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(300px, 1fr))",
                gap: 20,
                marginTop: 20,
                alignItems: "start",
              }}
            >
              {/* Оригинальная фотография + точки */}
              {previewUrl && (
                <figure style={{ margin: 0 }}>
                  <figcaption style={{ marginBottom: 8 }}>
                    Оригинал + пересечения
                  </figcaption>

                  <svg
                    viewBox={`0 0 ${gridResult.original_width} ${gridResult.original_height}`}
                    style={{
                      width: "100%",
                      height: "auto",
                      border: "1px solid #ccc",
                      display: "block",
                    }}
                  >
                    <image
                      href={previewUrl}
                      x={0}
                      y={0}
                      width={gridResult.original_width}
                      height={gridResult.original_height}
                    />

                    {gridResult.found &&
                      gridResult.grid_points_original.map(
                        ([x, y], index) => (
                          <circle
                            key={index}
                            cx={x}
                            cy={y}
                            r={Math.max(
                              3,
                              gridResult.original_width / 450
                            )}
                            fill="#00ff55"
                            stroke="#004422"
                            strokeWidth={1.5}
                          />
                        )
                      )}
                  </svg>
                </figure>
              )}

              {/* Выпрямленная фотография + линии */}
              {gridResult.rectified_png_b64 && (
                <figure style={{ margin: 0 }}>
                  <figcaption style={{ marginBottom: 8 }}>
                    Выпрямленная доска + сетка
                  </figcaption>

                  <svg
                    viewBox={`0 0 ${gridResult.rectified_side} ${gridResult.rectified_side}`}
                    style={{
                      width: "100%",
                      height: "auto",
                      border: "1px solid #ccc",
                      display: "block",
                    }}
                  >
                    <image
                      href={b64ToDataUrl(
                        gridResult.rectified_png_b64
                      )}
                      width={gridResult.rectified_side}
                      height={gridResult.rectified_side}
                    />

                    {/* Вертикальные линии */}
                    {gridResult.x_lines.map((x, i) => (
                      <line
                        key={`x-${i}`}
                        x1={x}
                        y1={0}
                        x2={x}
                        y2={gridResult.rectified_side}
                        stroke="#00d5ff"
                        strokeWidth={2}
                        strokeOpacity={0.8}
                      />
                    ))}

                    {/* Горизонтальные линии */}
                    {gridResult.y_lines.map((y, i) => (
                      <line
                        key={`y-${i}`}
                        x1={0}
                        y1={y}
                        x2={gridResult.rectified_side}
                        y2={y}
                        stroke="#00d5ff"
                        strokeWidth={2}
                        strokeOpacity={0.8}
                      />
                    ))}
                    {/* Распознанные номера поверх сетки */}
                    {ocrResult?.candidates.map((move, index) => {
                      const x = gridResult.x_lines[move.col];
                      const y = gridResult.y_lines[move.row];

                      return (
                        <g key={index} pointerEvents="none">
                          <text
                            x={x + 8}
                            y={y - 8}
                            fontSize={20}
                            fontWeight="bold"
                            fill={move.confidence >= 70 ? "#00a000" : "red"}
                            stroke="white"
                            strokeWidth={3}
                            paintOrder="stroke"
                          >
                            {move.number}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </figure>
              )}
            </div>
          </div>
        )}
      </section>
        {gridResult?.found && gridResult.rectified_png_b64 && (
          <section
            style={{
              marginTop: 24,
              borderTop: "2px solid #ddd",
              paddingTop: 16,
            }}
          >
            <h2>Распознавание ходов</h2>

            <button
              onClick={onRecognize}
              disabled={ocrProcessing || gridProcessing}
            >
              {ocrProcessing
                ? "Распознаём числа..."
                : "Распознать числа → SGF"}
            </button>

            {ocrResult && (
              <div style={{ marginTop: 20 }}>

                <p>
                  <b>Всего пересечений:</b>{" "}
                  {ocrResult.inspected_cells}
                  <br />

                  <b>OCR-вызовов:</b>{" "}
                  {ocrResult.ocr_attempts}
                  <br />

                  <b>Распознано чисел:</b>{" "}
                  {ocrResult.candidates.length}
                  <br />

                  <b>Ходов экспортировано:</b>{" "}
                  {ocrResult.exported_moves.length}
                </p>

                {ocrResult.warnings.length > 0 && (
                  <div style={{ color: "darkorange" }}>
                    <h3>Предупреждения</h3>

                    {ocrResult.warnings.map((warning, i) => (
                      <p key={i}>{warning}</p>
                    ))}
                  </div>
                )}

                {ocrResult.sgf && (
                  <button
                    onClick={() => downloadSgf(ocrResult.sgf!)}
                  >
                    Скачать SGF
                  </button>
                )}

                {!ocrResult.sgf && (
                  <p>
                    SGF пока не сформирован:
                    не удалось восстановить начало партии.
                  </p>
                )}

                <h3>Распознанные числа</h3>

                <table
                  style={{
                    borderCollapse: "collapse",
                    width: "100%",
                    maxWidth: 600,
                  }}
                >
                  <thead>
                    <tr>
                      <th style={cell}>Ход</th>
                      <th style={cell}>Строка</th>
                      <th style={cell}>Столбец</th>
                      <th style={cell}>Confidence</th>
                    </tr>
                  </thead>

                  <tbody>
                    {ocrResult.candidates.map((move, i) => (
                      <tr key={i}>
                        <td style={cell}>{move.number}</td>
                        <td style={cell}>{move.row}</td>
                        <td style={cell}>{move.col}</td>
                        <td style={cell}>
                          {move.confidence.toFixed(1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
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