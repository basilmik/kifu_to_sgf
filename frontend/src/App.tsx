import { useEffect, useState, } from "react";
import { b64ToDataUrl, debugDetect, 
  detectBoard, processImage, detectGrid, 
  recognizeGrid, compareGridOCR, compareDigitApproaches,
  extractGridCells, maskGridLines } from "./api";
import type {
  DebugDetectResponse,
  DetectBoardResponse,
  ProcessResponse,
  GridResponse,
  RecognitionResponse,
  OCRComparisonResponse,
  DigitComparisonResponse,
  ExtractCellsResponse,
  GridMaskResponse,
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
  
  const [comparing, setComparing] = useState(false);

  const [comparisonResult, setComparisonResult] = useState<OCRComparisonResponse | null>(null);

  const [digitComparing, setDigitComparing] = useState(false);

  const [digitComparison, setDigitComparison] =
    useState<DigitComparisonResponse | null>(null);

  const [cellsProcessing, setCellsProcessing] = useState(false);
  const [cellsResult, setCellsResult] =
    useState<ExtractCellsResponse | null>(null);

  const [maskProcessing, setMaskProcessing] = useState(false);
  const [maskResult, setMaskResult] =
    useState<GridMaskResponse | null>(null);
  const [maskMode, setMaskMode] = useState<"lighten" | "white">("lighten");
  const [maskWidth, setMaskWidth] = useState(5);
  const [maskLiftFactor, setMaskLiftFactor] = useState(0.85);

  // Необязательный выбор конкретного пересечения.
  const [testRow, setTestRow] = useState("");
  const [testCol, setTestCol] = useState("");

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
    setComparisonResult(null);
    setDigitComparison(null);
    setCellsResult(null);
    setMaskResult(null);
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
    setComparisonResult(null);
    setDigitComparison(null);
    setCellsResult(null);
    setMaskResult(null);
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

  async function onCompareOCR() {
    if (!gridResult?.found) return;

    setComparing(true);
    setComparisonResult(null);
    setError(null);

    try {
      const result = await compareGridOCR(gridResult);
      setComparisonResult(result);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setComparing(false);
    }
  }

  async function onCompareDigits() {
    if (!gridResult?.found) return;

    setDigitComparing(true);
    setDigitComparison(null);
    setError(null);

    try {
      let cells: [number, number][] | undefined;

      if (testRow !== "" && testCol !== "") {
        const row = Number(testRow);
        const col = Number(testCol);

        if (
          !Number.isInteger(row) ||
          !Number.isInteger(col) ||
          row < 0 ||
          col < 0 ||
          row >= gridResult.grid_size ||
          col >= gridResult.grid_size
        ) {
          throw new Error("Некорректные row/col");
        }

        cells = [[row, col]];
      }

      const result = await compareDigitApproaches(
        gridResult,
        cells
      );

      setDigitComparison(result);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setDigitComparing(false);
    }
  }

  async function onMaskGrid() {
    if (!gridResult?.found) return;

    setMaskProcessing(true);
    setMaskResult(null);
    // После пересчёта маски старые квадратики невалидны.
    setCellsResult(null);
    setError(null);

    try {
      const result = await maskGridLines(gridResult, {
        line_width: maskWidth,
        mode: maskMode,
        lift_factor: maskLiftFactor,
      });
      setMaskResult(result);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setMaskProcessing(false);
    }
  }

  async function onExtractCells() {
    if (!gridResult?.found || !maskResult) return;

    setCellsProcessing(true);
    setCellsResult(null);
    setError(null);

    try {
      // Режем именно изображение после маски.
      const result = await extractGridCells(
        maskResult.masked_png_b64,
        gridResult.grid_size,
        gridResult.x_lines,
        gridResult.y_lines,
      );
      setCellsResult(result);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setCellsProcessing(false);
    }
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

              {/* Выпрямленная доска без сетки */}
              {maskResult && (
                <figure style={{ margin: 0 }}>
                  <figcaption style={{ marginBottom: 8 }}>
                    Выпрямленная доска без сетки ({maskResult.mode})
                  </figcaption>

                  <img
                    src={b64ToDataUrl(maskResult.masked_png_b64)}
                    alt="masked"
                    style={{
                      width: "100%",
                      height: "auto",
                      border: "1px solid #ccc",
                      display: "block",
                    }}
                  />
                </figure>
              )}
            </div>

            <div
              style={{
                marginTop: 20,
                padding: 12,
                border: "1px solid #ddd",
                borderRadius: 6,
                display: "flex",
                flexWrap: "wrap",
                gap: 12,
                alignItems: "center",
              }}
            >
              <b>Осветлить сетку:</b>

              <label>
                режим{" "}
                <select
                  value={maskMode}
                  onChange={(e) =>
                    setMaskMode(e.target.value as "lighten" | "white")
                  }
                >
                  <option value="lighten">lighten (к белому)</option>
                  <option value="white">white (залить белым)</option>
                </select>
              </label>

              <label>
                толщина, px{" "}
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={maskWidth}
                  onChange={(e) =>
                    setMaskWidth(Number(e.target.value) || 1)
                  }
                  style={{ width: 70 }}
                />
              </label>

              <label>
                lift factor{" "}
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={maskLiftFactor}
                  onChange={(e) =>
                    setMaskLiftFactor(Number(e.target.value))
                  }
                  style={{ width: 70 }}
                />
              </label>

              <button
                onClick={onMaskGrid}
                disabled={maskProcessing || gridProcessing}
              >
                {maskProcessing ? "Обрабатываем..." : "Применить маску"}
              </button>

              <button
                onClick={onExtractCells}
                disabled={
                  !maskResult ||
                  cellsProcessing ||
                  maskProcessing ||
                  gridProcessing
                }
              >
                {cellsProcessing
                  ? "Вырезаем..."
                  : "Вырезать квадратики"}
              </button>
            </div>

            {cellsResult && (
              <div style={{ marginTop: 20 }}>
                <h3>Вырезанные квадратики</h3>
                <p>
                  <b>Размер сетки:</b>{" "}
                  {cellsResult.grid_size}×{cellsResult.grid_size}
                  <br />
                  <b>Вырезано:</b>{" "}
                  {cellsResult.cells.length}
                </p>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: `repeat(${cellsResult.grid_size}, minmax(0, 1fr))`,
                    gap: 2,
                    maxWidth: 900,
                    marginTop: 12,
                  }}
                >
                  {cellsResult.cells.map((c) => (
                    <figure
                      key={`${c.row}-${c.col}`}
                      style={{
                        margin: 0,
                        padding: 2,
                        border: "1px solid #ccc",
                        background: "#fafafa",
                      }}
                      title={`row=${c.row}, col=${c.col}`}
                    >
                      {c.png_b64 ? (
                        <img
                          src={b64ToDataUrl(c.png_b64)}
                          alt={`cell ${c.row}-${c.col}`}
                          style={{
                            width: "100%",
                            height: "auto",
                            display: "block",
                            imageRendering: "pixelated",
                          }}
                        />
                      ) : (
                        <div
                          style={{
                            width: "100%",
                            aspectRatio: "1 / 1",
                            background: "#eee",
                          }}
                        />
                      )}
                      <figcaption
                        style={{
                          fontSize: 10,
                          textAlign: "center",
                          color: "#666",
                          lineHeight: 1.25,
                        }}
                      >
                        <div>
                          {c.row},{c.col}
                        </div>
                        <div>
                          {c.height}×{c.width}px
                        </div>
                        <div>
                          {c.mnist_digit !== null
                            ? `${c.mnist_digit} (${c.mnist_confidence!.toFixed(2)})`
                            : "—"}
                        </div>
                      </figcaption>
                    </figure>
                  ))}
                </div>
              </div>
            )}
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

        {gridResult?.found && (
          <section
            style={{
              marginTop: 24,
              paddingTop: 16,
              borderTop: "2px solid #ddd",
            }}
          >
            <h2>Сравнение OCR-моделей</h2>

            <button
              onClick={onCompareOCR}
              disabled={comparing || gridProcessing}
            >
              {comparing
                ? "Сравниваем модели..."
                : "Сравнить Tesseract и PaddleOCR"}
            </button>

            {comparisonResult && (
              <div style={{ marginTop: 20 }}>
                <p>
                  Проверено пересечений:{" "}
                  {comparisonResult.inspected_cells}
                  <br />
                  Передано в OCR:{" "}
                  {comparisonResult.ocr_attempts}
                </p>

                <p style={{ color: "#666" }}>
                  Сравниваются одни и те же фрагменты.
                  Результаты пока не используются для SGF.
                </p>

                <div style={{ overflowX: "auto" }}>
                  <table
                    style={{
                      borderCollapse: "collapse",
                      width: "100%",
                    }}
                  >
                    <thead>
                      <tr>
                        <th style={cell}>ROI</th>
                        <th style={cell}>Позиция</th>
                        <th style={cell}>Tesseract</th>
                        <th style={cell}>PaddleOCR</th>
                      </tr>
                    </thead>

                    <tbody>
                      {comparisonResult.comparisons.map(
                        (item) => {
                          const different =
                            item.tesseract.text !==
                            item.paddle.text;

                          return (
                            <tr
                              key={`${item.row}-${item.col}`}
                              style={{
                                background: different
                                  ? "#fff5e6"
                                  : "transparent",
                              }}
                            >
                              <td style={cell}>
                                <img
                                  src={b64ToDataUrl(
                                    item.roi_png_b64
                                  )}
                                  alt="OCR crop"
                                  style={{
                                    width: 90,
                                    maxWidth: "none",
                                  }}
                                />
                              </td>

                              <td style={cell}>
                                ({item.row}, {item.col})
                              </td>

                              <td style={cell}>
                                <b>
                                  {item.tesseract.text || "—"}
                                </b>
                                <div>
                                  {item.tesseract.confidence.toFixed(1)}
                                </div>
                              </td>

                              <td style={cell}>
                                <b>
                                  {item.paddle.text || "—"}
                                </b>
                                <div>
                                  {item.paddle.confidence.toFixed(1)}
                                </div>
                              </td>
                            </tr>
                          );
                        }
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        )}
        {gridResult?.found && (
          <section style={{ marginTop: 24 }}>
            <h2>MNIST vs CRNN</h2>

            <p>
              По умолчанию сравниваются 6 ROI с наибольшим
              количеством обнаруженных штрихов.
              Для проверки конкретной точки укажи row и col
              (индексация с нуля).
            </p>

            <div style={{
              display: "flex",
              gap: 8,
              marginBottom: 12,
            }}>
              <input
                type="number"
                placeholder="row"
                value={testRow}
                onChange={(e) => setTestRow(e.target.value)}
                style={{ width: 80 }}
              />

              <input
                type="number"
                placeholder="col"
                value={testCol}
                onChange={(e) => setTestCol(e.target.value)}
                style={{ width: 80 }}
              />
            </div>

            <button
              disabled={digitComparing || gridProcessing}
              onClick={onCompareDigits}
            >
              {digitComparing
                ? "Сравниваем..."
                : "Сравнить MNIST и CRNN"}
            </button>

            {digitComparison && (
              <>
                <p>
                  Время обработки:{" "}
                  {(digitComparison.elapsed_ms / 1000).toFixed(2)} с
                </p>

                <div style={{ overflowX: "auto" }}>
                  <table style={{
                    width: "100%",
                    borderCollapse: "collapse",
                  }}>
                    <thead>
                      <tr>
                        <th style={cell}>ROI</th>
                        <th style={cell}>Сегментация</th>
                        <th style={cell}>MNIST</th>
                        <th style={cell}>CRNN</th>
                      </tr>
                    </thead>

                    <tbody>
                      {digitComparison.comparisons.map((item) => (
                        <tr key={`${item.row}-${item.col}`}>
                          <td style={cell}>
                            <img
                              src={b64ToDataUrl(item.roi_png_b64)}
                              alt="Original ROI"
                              width={90}
                            />
                            <div>
                              ({item.row}, {item.col})
                            </div>
                          </td>

                          <td style={cell}>
                            <img
                              src={b64ToDataUrl(
                                item.segmentation_png_b64
                              )}
                              alt="Segmentation"
                              width={90}
                            />
                            <div style={{ display: "flex", gap: 4 }}>
                              {item.digit_png_b64.map((png, i) => (
                                <img
                                  key={i}
                                  src={b64ToDataUrl(png)}
                                  alt={`Digit ${i}`}
                                  width={36}
                                  style={{ imageRendering: "pixelated" }}
                                />
                              ))}
                            </div>
                            <small>{item.mnist.status}</small>
                          </td>

                          <td style={cell}>
                            <b>{item.mnist.text ?? "—"}</b>
                            <div>
                              {item.mnist.confidence_per_digit
                                .map((v) => v.toFixed(2))
                                .join(", ")}
                            </div>
                          </td>

                          <td style={cell}>
                            <b>{item.crnn.text || "—"}</b>
                            <div>
                              {item.crnn.valid_number
                                ? "Числовая строка"
                                : "Не число / пусто"}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
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