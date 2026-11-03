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

export interface DebugContour {
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
  top_contours: DebugContour[];
}

export interface GridResponse {
  found: boolean;
  note: string;
  grid_size: number;
  original_width: number;
  original_height: number;
  rectified_side: number;
  rectified_png_b64: string | null;
  corners_original: [number, number][];
  grid_points_original: [number, number][];
  x_lines: number[];
  y_lines: number[];
  x_quality: number;
  y_quality: number;
}

export interface RecognitionCandidate {
  number: number;
  row: number;
  col: number;
  confidence: number;
}

export interface RecognitionResponse {
  candidates: RecognitionCandidate[];
  inspected_cells: number;
  ocr_attempts: number;
  exported_moves: RecognitionCandidate[];
  sgf: string | null;
  warnings: string[];
}

export interface OCRComparisonItem {
  row: number;
  col: number;
  ink_score: number;
  roi_png_b64: string;
  tesseract: { text: string; confidence: number };
  paddle: { text: string; confidence: number };
}

export interface OCRComparisonResponse {
  comparisons: OCRComparisonItem[];
  inspected_cells: number;
  ocr_attempts: number;
}

export interface DigitComparisonItem {
  row: number;
  col: number;
  ink_score: number;
  roi_png_b64: string;
  segmentation_png_b64: string;
  digit_png_b64: string[];
  mnist: {
    text: string | null;
    confidence_per_digit: number[];
    status: string;
  };
  crnn: {
    text: string;
    valid_number: boolean;
  };
}

export interface DigitComparisonResponse {
  comparisons: DigitComparisonItem[];
  elapsed_ms: number;
}

export interface ExtractCellsResponse {
  grid_size: number;
  cells: {
    row: number;
    col: number;
    width: number;
    height: number;
    png_b64: string;
    mnist_digit: string | null;
    mnist_confidence: number | null;
  }[];
}

export interface GridMaskResponse {
  width: number;
  height: number;
  masked_png_b64: string;
  mode: string;
  line_width: number;
  dim_factor: number;
}