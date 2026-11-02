import type {
  DebugDetectResponse,
  DetectBoardResponse,
  ProcessResponse,
  RecognitionResponse,
  GridResponse,
} from "./types";

export async function processImage(file: File): Promise<ProcessResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/process", { method: "POST", body: form });
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}

export async function detectBoard(file: File): Promise<DetectBoardResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/detect-board", { method: "POST", body: form });
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}

export async function debugDetect(file: File): Promise<DebugDetectResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/debug-detect", { method: "POST", body: form });
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}

export function b64ToDataUrl(b64: string, mime = "image/png"): string {
  return `data:${mime};base64,${b64}`;
}

async function extractError(res: Response): Promise<string> {
  let detail = `HTTP ${res.status}`;
  try {
    const body = await res.json();
    if (body && body.detail) detail = body.detail;
  } catch {
    // ignore
  }
  return detail;
}

export async function detectGrid(
  file: File
): Promise<GridResponse> {
  const form = new FormData();

  form.append("file", file);
  form.append("grid_size", "19");

  const response = await fetch("/api/grid", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Grid detection failed: ${text}`);
  }

  return response.json() as Promise<GridResponse>;
}

export async function recognizeGrid(
  grid: GridResponse
): Promise<RecognitionResponse> {

  if (!grid.rectified_png_b64) {
    throw new Error("Нет выпрямленного изображения");
  }

  const response = await fetch(
    "/api/recognize-sgf",
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        rectified_png_b64: grid.rectified_png_b64,

        grid_size: grid.grid_size,

        x_lines: grid.x_lines,
        y_lines: grid.y_lines,

        max_move: 500,
      }),
    }
  );

  if (!response.ok) {
    const text = await response.text();

    throw new Error(
      `Recognition failed (${response.status}): ${text}`
    );
  }

  return response.json() as Promise<RecognitionResponse>;
}