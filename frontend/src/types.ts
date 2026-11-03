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

export type GridPoint = [number, number];

export interface GridResponse {
  found: boolean;
  note: string;

  grid_size: number;

  original_width: number;
  original_height: number;

  rectified_side: number;
  rectified_png_b64: string | null;

  corners_original: GridPoint[];
  grid_points_original: GridPoint[];

  x_lines: number[];
  y_lines: number[];

  x_quality: number;
  y_quality: number;
}

export interface RecognizedMove {
  number: number;
  row: number;
  col: number;
  confidence: number;
}

export interface RecognitionResponse {
  candidates: RecognizedMove[];

  inspected_cells: number;
  ocr_attempts: number;

  exported_moves: RecognizedMove[];

  sgf: string | null;

  warnings: string[];
}

export interface OCRPrediction {
  text: string;
  confidence: number;
}

export interface OCRComparison {
  row: number;
  col: number;
  ink_score: number;

  roi_png_b64: string;

  tesseract: OCRPrediction;
  paddle: OCRPrediction;
}

export interface OCRComparisonResponse {
  comparisons: OCRComparison[];
  inspected_cells: number;
  ocr_attempts: number;
}

export interface DigitComparison {
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
  comparisons: DigitComparison[];
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
