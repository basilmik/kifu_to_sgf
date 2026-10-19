import type { ProcessResponse } from "./types";

export async function processImage(file: File): Promise<ProcessResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/api/process", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  return res.json();
}

export function b64ToDataUrl(b64: string, mime = "image/png"): string {
  return `data:${mime};base64,${b64}`;
}