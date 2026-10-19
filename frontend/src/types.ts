export interface ProcessResponse {
  width: number;
  height: number;
  original_png_b64: string;
  processed_png_b64: string;
}

export interface DetectBoardResponse {
  found: boolean;
  width: number;
  height: number;
  original_png_b64: string;
  marked_png_b64: string;
  corners: [number, number][];
}