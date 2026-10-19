import type {
  DebugDetectResponse,
  DetectBoardResponse,
  ProcessResponse,
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