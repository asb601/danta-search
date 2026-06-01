import type { Metadata } from "next";
import { Instrument_Serif, Syne } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

/* ── Fonts ──────────────────────────────────────────────────────────────────
   Instrument Serif  — editorial optical serif; almost zero usage in AI tools
   Syne              — geometric sans with character; distinctive at every size
─────────────────────────────────────────────────────────────────────────── */
const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
  weight: "400",
  style: ["normal", "italic"],
});

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
  weight: ["400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "danta-search",
  description: "Enterprise Data Intelligence — Chat with your data at scale",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${instrumentSerif.variable} ${syne.variable}`}>
      <body className="min-h-screen">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
