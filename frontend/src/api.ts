import type { SumRequest, SumResponse } from "./types";

export async function sum(req: SumRequest): Promise<SumResponse> {
  const res = await fetch("/api/sum", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}