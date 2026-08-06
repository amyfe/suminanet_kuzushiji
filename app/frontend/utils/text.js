// Mirrors KURONET_CER_SCORE_THRESH (0.35, the cutoff for a char surviving
// detection at all) and TRANSLATION_UNCERTAIN_SCORE_THRESH (0.6, the "still
// shaky" bar used to bracket chars for Claude) in config.py, so the UI's
// confidence tiers stay conceptually aligned with the backend's own bars.
const SCORE_LOW_THRESH = 0.35
const SCORE_MEDIUM_THRESH = 0.6

export function scoreTier(score) {
  if (score < SCORE_LOW_THRESH) return 'low'
  if (score < SCORE_MEDIUM_THRESH) return 'medium'
  return 'high'
}

export function formatScore(score) {
  return (score * 100).toFixed(1) + '%'
}

export function columnsOf(text, maxCols) {
  const chars = Array.from(text)
  const numCols = Math.min(maxCols, Math.max(1, Math.ceil(chars.length / 3)))
  const perCol = Math.ceil(chars.length / numCols)
  const groups = []
  for (let i = 0; i < chars.length; i += perCol) groups.push(chars.slice(i, i + perCol))
  return groups.reverse()
}

function median(nums) {
  const sorted = [...nums].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

// Group per-char boxes (already in model reading order: right-to-left
// columns, top-to-bottom within each column) into their true source columns
// by x-center proximity, mirroring the backend's column clustering
// (ROIReadingOrder._vertical_sort_indices in roi_ordering.py). Unlike
// columnsOf(), this respects the manuscript's actual column boundaries
// instead of chopping the flat string into fixed-length blocks.
// Returns column groups of char objects, in reading order (first = rightmost).
export function columnsFromChars(chars) {
  if (chars.length === 0) return []
  const widths = chars.map((c) => c.box[2] - c.box[0])
  const gapThresh = Math.max(median(widths) * 1.5, 5)

  const cols = []
  let cur = [chars[0]]
  let curXCenter = (chars[0].box[0] + chars[0].box[2]) / 2
  for (let i = 1; i < chars.length; i++) {
    const c = chars[i]
    const xc = (c.box[0] + c.box[2]) / 2
    if (Math.abs(xc - curXCenter) > gapThresh) {
      cols.push(cur)
      cur = []
      curXCenter = xc
    } else {
      curXCenter = (curXCenter * cur.length + xc) / (cur.length + 1)
    }
    cur.push(c)
  }
  cols.push(cur)
  return cols
}
