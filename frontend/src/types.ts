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

export interface DebugImage {
  name: string;
  width: number;
  height: number;
  png_b64: string;
}

export interface ContourInfo {
  area: number;
  perimeter: number;
  approx_points: number;
  reasonable: boolean;
  points: [number, number][];
}

export interface DebugDetectResponse {
  found: boolean;
  note: string;
  img_area: number;
  width: number;
  height: number;
  corners: [number, number][];
  images: DebugImage[];
  top_contours: ContourInfo[];
}