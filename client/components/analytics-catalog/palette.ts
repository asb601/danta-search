// Maps chart series to the OKLch design tokens already defined in globals.css
// (--chart-1 .. --chart-5). Pure-CSS-variable references keep theming centralized.

export const CHART_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];

export function colorAt(index: number): string {
  return CHART_COLORS[index % CHART_COLORS.length];
}

// Headline tiles (KPI value, gauge center, progress value, ranked-bar labels)
// must stay legible: a 12,345,678 spend reads as "$12.3M", not a long digit run.
// We compact magnitudes at/above 100,000 with K/M/B suffixes; smaller values keep
// full precision so e.g. counts and rates are exact.
const _COMPACT_FLOOR = 100_000;

// Number / currency / percent formatting shared across catalog components.
export function formatValue(
  value: unknown,
  format: "currency" | "percent" | "number" | "auto" = "auto",
): string {
  if (value === null || value === undefined || value === "") return "—";
  const num = typeof value === "number" ? value : Number(value);
  if (Number.isNaN(num)) return String(value);

  if (format === "currency") {
    if (Math.abs(num) >= _COMPACT_FLOOR) {
      return Intl.NumberFormat(undefined, {
        style: "currency",
        currency: "USD",
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(num);
    }
    return num.toLocaleString(undefined, {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: Math.abs(num) >= 1000 ? 0 : 2,
    });
  }
  if (format === "percent") {
    return `${num.toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
  }
  // number / auto — compact large magnitudes so headline tiles stay readable.
  if (Math.abs(num) >= _COMPACT_FLOOR) {
    return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(num);
  }
  return num.toLocaleString(undefined, {
    maximumFractionDigits: Number.isInteger(num) ? 0 : 2,
  });
}

export function compactNumber(value: number): string {
  return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}
