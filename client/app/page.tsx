"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search, BarChart3, Zap, ShieldCheck, ArrowRight,
} from "lucide-react";

/* ─── Typing hero ─── */
const QUERIES = [
  "What were the top 5 SKUs by revenue last quarter?",
  "Show me purchase orders above ₹10L from April.",
  "Which vendors have outstanding invoices over 60 days?",
  "Compare Q1 vs Q2 sales by region.",
  "Summarise the FBL3N report for finance.",
];

function TypingHero() {
  const [queryIdx, setQueryIdx] = useState(0);
  const [displayed, setDisplayed] = useState("");
  const [phase, setPhase] = useState<"typing" | "pause" | "erasing">("typing");

  useEffect(() => {
    const target = QUERIES[queryIdx];
    let timeout: ReturnType<typeof setTimeout>;

    if (phase === "typing") {
      if (displayed.length < target.length) {
        timeout = setTimeout(() => setDisplayed(target.slice(0, displayed.length + 1)), 38);
      } else {
        timeout = setTimeout(() => setPhase("pause"), 1800);
      }
    } else if (phase === "pause") {
      timeout = setTimeout(() => setPhase("erasing"), 400);
    } else {
      if (displayed.length > 0) {
        timeout = setTimeout(() => setDisplayed(displayed.slice(0, -1)), 18);
      } else {
        setQueryIdx((i) => (i + 1) % QUERIES.length);
        setPhase("typing");
      }
    }
    return () => clearTimeout(timeout);
  }, [displayed, phase, queryIdx]);

  return (
    <span className="text-[color:var(--fg)]">
      {displayed}
      <span className="cursor-blink" />
    </span>
  );
}

/* ─── Feature tabs ─── */
const FEATURES = [
  {
    id: "analytics",
    label: "Business Analytics",
    icon: BarChart3,
    title: "Understands your business logic",
    body: "Column semantics, KPIs, date ranges, fiscal calendars — danta-search learns how your business thinks, not just what your data contains.",
    preview: (
      <div className="p-5 flex flex-col gap-2.5">
        {[["Total Revenue", "₹1.24Cr", "+14.2%", true], ["Orders Shipped", "8,430", "+7.8%", true], ["Avg Order Value", "₹14,700", "−2.1%", false], ["Active Vendors", "142", "+3.4%", true]].map(([label, val, chg, up]) => (
          <div key={String(label)} className="flex items-center justify-between py-2 border-b border-[#e5e5e5] last:border-0">
            <span className="text-[12px] text-[#737373]">{label}</span>
            <div className="flex items-center gap-3">
              <span className="text-[13px] font-semibold text-[#0a0a0a]">{val}</span>
              <span className={`text-[11px] font-semibold ${up ? "text-[#16a34a]" : "text-[#dc2626]"}`}>{String(chg)}</span>
            </div>
          </div>
        ))}
      </div>
    ),
  },
  {
    id: "performance",
    label: "GB-Scale Speed",
    icon: Zap,
    title: "Millions of rows, instant answers",
    body: "Apache DataFusion executes Parquet scans with predicate pushdown across your entire data estate. No sampling. No slowdowns.",
    preview: (
      <div className="p-5 flex flex-col gap-3">
        <div className="flex items-end gap-1.5 h-20">
          {[35, 55, 42, 78, 60, 88, 65, 92, 70, 85, 95, 82].map((h, i) => (
            <motion.div
              key={i}
              initial={{ height: 0 }}
              animate={{ height: `${h}%` }}
              transition={{ delay: i * 0.04, duration: 0.5, ease: "easeOut" }}
              className="flex-1 rounded-sm"
              style={{ background: i >= 10 ? "#0a0a0a" : i >= 8 ? "rgba(10,10,10,0.45)" : "rgba(10,10,10,0.12)" }}
            />
          ))}
        </div>
        <div className="flex justify-between text-[10px] text-[#a3a3a3]">
          <span>Jan</span><span>Mar</span><span>Jun</span><span>Sep</span><span>Dec</span>
        </div>
        <div className="flex gap-4 pt-1">
          {[["280k rows", "scanned"], ["0.38s", "response"], ["99.9%", "uptime"]].map(([n, l]) => (
            <div key={l}>
              <div className="text-[15px] font-bold text-[#0a0a0a]">{n}</div>
              <div className="text-[10px] text-[#a3a3a3]">{l}</div>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    id: "retrieval",
    label: "Hybrid Search",
    icon: Search,
    title: "Finds what you mean, not just what you wrote",
    body: "OpenSearch hybrid (BM25 + vector) with native RRF ranking. Accurate results across both structured tables and unstructured document content.",
    preview: (
      <div className="p-5 space-y-2">
        {[
          { q: "overdue invoices south region", file: "FBL3N_report.pdf", score: 96 },
          { q: "purchase orders Q2 vendor", file: "purchase_orders_2025.xlsx", score: 89 },
          { q: "SKU revenue last quarter", file: "sales_q3_south.xlsx", score: 84 },
        ].map((r) => (
          <div key={r.file} className="flex items-center gap-3 p-2.5 rounded-lg bg-[#f9f9f9] border border-[#e5e5e5]">
            <div className="w-7 h-7 rounded-md bg-[#0a0a0a]/08 flex items-center justify-center">
              <Search className="w-3 h-3 text-[#0a0a0a]" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[11.5px] font-medium text-[#0a0a0a] truncate">{r.file}</div>
              <div className="text-[10px] text-[#a3a3a3] truncate">{r.q}</div>
            </div>
            <div className="text-[11px] font-bold text-[#0a0a0a]">{r.score}%</div>
          </div>
        ))}
      </div>
    ),
  },
  {
    id: "trust",
    label: "Source Citations",
    icon: ShieldCheck,
    title: "Every answer is fully auditable",
    body: "Results link back to the exact file, sheet, and row. Finance and compliance teams get the paper trail they need without asking for it.",
    preview: (
      <div className="p-5 space-y-3">
        <div className="text-[12.5px] text-[#0a0a0a] leading-relaxed">SKU-4821 generated <strong>₹28.4L</strong> in Q3 across South region.</div>
        <div className="space-y-1.5">
          {[{ file: "sales_q3_south.xlsx", rows: "rows 1,203–5,421", sheet: "Sheet1" }, { file: "sales_q3_north.xlsx", rows: "rows 880–2,100", sheet: "Q3 Data" }].map((s) => (
            <div key={s.file} className="flex items-center gap-2.5 px-3 py-2 rounded-lg border border-[#e5e5e5] bg-[#f9f9f9]">
              <div className="w-2 h-2 rounded-full bg-[#16a34a]" />
              <div>
                <div className="text-[11px] font-semibold text-[#0a0a0a] font-mono">{s.file}</div>
                <div className="text-[10px] text-[#a3a3a3]">{s.rows} · {s.sheet}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    ),
  },
];

/* ─── Black pill nav ─── */
const NAV_LINKS = ["Product", "Solutions", "Pricing", "Blog"];

function NavPill() {
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <nav className="hidden md:flex">
      <motion.div
        initial={{ opacity: 0, y: -6, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ delay: 0.15, duration: 0.4, ease: "easeOut" }}
        className="flex items-center gap-0.5 bg-[#0a0a0a] rounded-full px-1.5 py-1.5"
        style={{ boxShadow: "0 2px 12px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.06)" }}
      >
        {NAV_LINKS.map((label, i) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.22 + i * 0.055, duration: 0.3 }}
            className="relative"
            onHoverStart={() => setHovered(label)}
            onHoverEnd={() => setHovered(null)}
          >
            {/* White glow highlight on hover */}
            <AnimatePresence>
              {hovered === label && (
                <motion.span
                  key="glow"
                  layoutId="nav-hover-glow"
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ type: "spring", stiffness: 380, damping: 28 }}
                  className="absolute inset-0 rounded-full bg-white/[0.12]"
                  style={{ boxShadow: "0 0 12px rgba(255,255,255,0.1)" }}
                />
              )}
            </AnimatePresence>
            <Link
              href="#"
              className="relative z-10 flex items-center px-3.5 py-1.5 text-[13px] font-medium text-white/70 hover:text-white transition-colors duration-150 rounded-full select-none"
            >
              {label}
            </Link>
          </motion.div>
        ))}
      </motion.div>
    </nav>
  );
}

/* ─── Main page ─── */
export default function HomePage() {
  const [activeTab, setActiveTab] = useState("analytics");
  const activeFeature = FEATURES.find((f) => f.id === activeTab)!;

  return (
    <div className="flex flex-col min-h-screen bg-white text-[color:var(--fg)]">

      {/* ── NAV ── */}
      <motion.header
        initial={{ y: -12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
        className="nav-bar"
      >
        <div className="page-container flex items-center justify-between h-14 px-4 sm:px-6">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 shrink-0">
            <div className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0" style={{ backgroundColor: "var(--fg)" }}>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <circle cx="6" cy="6" r="3.5" stroke="var(--bg)" strokeWidth="1.5" />
                <circle cx="6" cy="6" r="1.25" fill="var(--bg)" />
              </svg>
            </div>
            <span className="text-[14px] font-semibold tracking-tight">
              danta-search
            </span>
          </Link>

          {/* Center black pill nav */}
          <NavPill />

          {/* Right actions */}
          <div className="flex items-center gap-2 sm:gap-3">
            <Link href="/login" className="nav-link hidden sm:inline text-[13px]">Sign in</Link>
            <Link href="/login" className="btn-black px-3 sm:px-4 h-8 rounded-lg text-[12.5px] sm:text-[13px]">
              Book a Demo
            </Link>
          </div>
        </div>
      </motion.header>

      <main className="flex-1">

        {/* ── HERO ── */}
        <section className="relative overflow-hidden pt-14 pb-10 sm:pt-20 sm:pb-12 md:pt-32 md:pb-16 px-4 sm:px-6">
          {/* Grid background — fades at bottom */}
          <div
            className="absolute inset-0 grid-bg pointer-events-none"
            style={{
              opacity: 0.4,
              maskImage: "radial-gradient(ellipse 80% 55% at 50% 0%, black 30%, transparent 100%)",
            }}
          />

          <div className="relative page-container text-center">
            <motion.div
              initial="hidden"
              animate="show"
              variants={{ hidden: {}, show: { transition: { staggerChildren: 0.09 } } }}
              className="flex flex-col items-center"
            >
              {/* Label */}
              <motion.span
                variants={{ hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { duration: 0.5 } } }}
                className="section-label mb-5 inline-block"
              >
                Enterprise Data Intelligence
              </motion.span>

              {/* Main headline */}
              <motion.h1
                variants={{ hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0, transition: { duration: 0.6 } } }}
                className="display-xl mb-4 max-w-3xl"
              >
                Your enterprise data,
                <br />
                <span className="gradient-text">answered instantly.</span>
              </motion.h1>

              {/* Sub headline */}
              <motion.p
                variants={{ hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { duration: 0.55 } } }}
                className="body-lead max-w-[460px] mb-10"
              >
                Ask business questions across GBs of spreadsheets, PDFs, and documents in plain English.
              </motion.p>

              {/* CTA buttons */}
              <motion.div
                variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { duration: 0.5 } } }}
                className="flex flex-col xs:flex-row gap-3 items-center mb-8 sm:mb-10 w-full sm:w-auto"
              >
                <Link href="/login" className="btn-black px-5 sm:px-6 h-11 rounded-xl gap-2 w-full sm:w-auto justify-center">
                  Get Started Free <ArrowRight className="w-4 h-4" />
                </Link>
                <Link href="#" className="btn-outline px-5 sm:px-6 h-11 rounded-xl w-full sm:w-auto justify-center">
                  See a Demo
                </Link>
              </motion.div>

              {/* Live typing box */}
              <motion.div
                variants={{ hidden: { opacity: 0, scale: 0.97 }, show: { opacity: 1, scale: 1, transition: { duration: 0.55 } } }}
                className="w-full max-w-xl bg-white border border-[#e5e5e5] rounded-xl px-5 py-3.5 text-left shadow-sm flex items-center gap-3"
              >
                <Search className="w-4 h-4 text-[#a3a3a3] shrink-0" />
                <span className="text-[14px] leading-relaxed min-h-[22px]">
                  <TypingHero />
                </span>
              </motion.div>

              <motion.p
                variants={{ hidden: { opacity: 0 }, show: { opacity: 1, transition: { duration: 0.5, delay: 0.2 } } }}
                className="mt-4 section-label"
              >
                Handles files up to{" "}
                <span className="text-[color:var(--fg)]">10 GB+</span>
                {" "}· Excel · CSV · PDF · DOCX
              </motion.p>
            </motion.div>
          </div>
        </section>

        {/* ── PRODUCT SHOWCASE ── */}
        <section className="px-4 sm:px-6 pb-14 sm:pb-20 page-container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            className="product-window"
          >
            {/* Window chrome */}
            <div className="window-chrome">
              <div className="window-dot bg-[#ff5f57]" />
              <div className="window-dot bg-[#febc2e]" />
              <div className="window-dot bg-[#28c840]" />
              <span className="ml-3 text-[11.5px] text-white/30 font-medium tracking-wider hidden sm:inline">
                danta-search · analytics
              </span>
            </div>

            {/* On mobile: stacked. On sm+: side by side */}
            <div className="bg-[#111111] flex flex-col sm:grid sm:[grid-template-columns:180px_1fr]" style={{ minHeight: 320 }}>
              {/* File list — hidden on very small, shown from sm */}
              <div className="hidden sm:block border-r border-white/[0.06] p-4">
                <p className="section-label text-white/30 mb-3">Files</p>
                {[
                  { ext: "XL", name: "sales_q3_south.xlsx", sub: "142k rows", color: "#4ade80", bg: "rgba(74,222,128,0.12)", active: true },
                  { ext: "XL", name: "purchase_orders.xlsx", sub: "89k rows", color: "#4ade80", bg: "rgba(74,222,128,0.12)", active: false },
                  { ext: "PDF", name: "FBL3N_report.pdf", sub: "312 pages", color: "#f87171", bg: "rgba(248,113,113,0.12)", active: false },
                  { ext: "CSV", name: "inventory.csv", sub: "220k rows", color: "#60a5fa", bg: "rgba(96,165,250,0.12)", active: false },
                ].map((f) => (
                  <div
                    key={f.name}
                    className={`flex items-center gap-2 px-2 py-2 rounded-lg mb-0.5 ${f.active ? "bg-white/[0.07]" : "hover:bg-white/[0.04]"}`}
                  >
                    <span className="w-7 h-7 rounded-md flex items-center justify-center text-[9px] font-bold shrink-0" style={{ background: f.bg, color: f.color }}>
                      {f.ext}
                    </span>
                    <div className="min-w-0">
                      <p className="text-[11px] text-white/70 font-medium leading-tight truncate">{f.name}</p>
                      <p className="text-[10px] text-white/25">{f.sub}</p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Chat */}
              <div className="p-4 sm:p-5 flex flex-col gap-3">
                <div className="flex justify-end">
                  <div className="bg-white/10 text-white/85 text-[12px] sm:text-[13px] px-3 sm:px-4 py-2.5 rounded-2xl rounded-tr-sm max-w-[85%] sm:max-w-[68%]">
                    What were the top 5 SKUs by revenue last quarter across all regions?
                  </div>
                </div>
                <div className="bg-white/[0.04] border border-white/[0.06] rounded-2xl rounded-tl-sm p-3 sm:p-4 max-w-[95%] sm:max-w-[90%]">
                  <p className="text-[11px] sm:text-[12px] text-white/55 mb-3">Based on Q3 2025 sales data (3 files, 280k rows):</p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-[11px] sm:text-[11.5px] border-collapse">
                      <thead>
                        <tr>
                          {["#", "SKU", "Region", "Revenue"].map((h) => (
                            <th key={h} className="text-left py-1.5 px-2 text-[9.5px] sm:text-[10px] font-semibold tracking-widest text-white/25 uppercase border-b border-white/[0.05]">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {[["01", "SKU-4821", "South", "₹28.4L"], ["02", "SKU-0093", "North", "₹21.1L"], ["03", "SKU-7712", "South", "₹18.9L"]].map(([r, s, reg, rev]) => (
                          <tr key={s}>
                            <td className="py-1.5 px-2 text-white/25">{r}</td>
                            <td className="py-1.5 px-2 text-white/85 font-semibold">{s}</td>
                            <td className="py-1.5 px-2 text-white/45">{reg}</td>
                            <td className="py-1.5 px-2 text-white/85 font-bold">{rev}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {["sales_q3_south.xlsx", "sales_q3_north.xlsx"].map((s) => (
                      <span key={s} className="text-[10px] bg-white/[0.07] px-2 py-0.5 rounded font-mono text-white/30">{s}</span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Input bar */}
            <div className="bg-[#111111] border-t border-white/[0.06] px-4 sm:px-5 py-3 flex items-center gap-3">
              <span className="text-[12px] sm:text-[13px] text-white/18 flex-1">Ask anything about your data…</span>
              <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: "rgba(255,255,255,0.08)" }}>
                <ArrowRight className="w-3.5 h-3.5 text-white/40" />
              </div>
            </div>
          </motion.div>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section className="page-section">
          <div className="page-container">
            <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6 mb-14">
              <div>
                <motion.span
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  className="section-label block mb-4"
                >
                  How it works
                </motion.span>
                <motion.h2
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.55, ease: "easeOut" }}
                  className="display-md"
                >
                  From upload to insight<br />in minutes.
                </motion.h2>
              </div>
              <p className="body-lead max-w-xs md:text-right">
                No SQL. No data engineering. No dashboards to configure.
              </p>
            </div>

            <div className="border border-[#e5e5e5] rounded-2xl overflow-hidden grid grid-cols-1 md:grid-cols-3">
              {[
                { n: "01", title: "Upload your files", body: "Excel, CSV, PDF, DOCX — any format, any size up to 10 GB+. No preprocessing required." },
                { n: "02", title: "We index & understand", body: "The platform learns your column semantics, metrics, date ranges, and cross-file relationships." },
                { n: "03", title: "Ask in plain English", body: "Get grounded, source-cited answers with tables and breakdowns — linked to the exact rows." },
              ].map((s, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: i * 0.1, ease: "easeOut" }}
                  className="step-card"
                >
                  <div className="step-number">{s.n}</div>
                  <h3 className="text-[15px] font-semibold mb-2 text-[#0a0a0a]">{s.title}</h3>
                  <p className="text-[13.5px] text-[#737373] leading-relaxed">{s.body}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* ── ANIMATED FEATURE TABS ── */}
        <section className="page-section">
          <div className="page-container">
            <motion.span
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="section-label block mb-4"
            >
              Capabilities
            </motion.span>
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.55 }}
              className="display-md mb-10 max-w-lg"
            >
              Built for enterprise-grade data at real scale.
            </motion.h2>

            {/* Tab bar — scrollable on mobile */}
            <div className="overflow-x-auto pb-1 mb-6 sm:mb-8 -mx-4 sm:mx-0 px-4 sm:px-0">
              <div className="relative flex items-center gap-1 p-1 bg-[#f4f4f4] rounded-full w-fit">
                {FEATURES.map((f) => (
                  <button
                    key={f.id}
                    onClick={() => setActiveTab(f.id)}
                    className={`feature-tab ${activeTab === f.id ? "active" : ""}`}
                  >
                    {activeTab === f.id && (
                      <motion.span
                        layoutId="feature-pill"
                        className="absolute inset-0 bg-white rounded-full shadow-sm border border-[#e5e5e5]"
                        transition={{ type: "spring", stiffness: 400, damping: 34 }}
                      />
                    )}
                    <span className="relative z-10 flex items-center gap-1.5">
                      <f.icon className="w-3.5 h-3.5" />
                      <span className="hidden xs:inline">{f.label}</span>
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Tab content */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.28, ease: "easeOut" }}
                className="grid md:grid-cols-2 gap-0 items-stretch border border-[#e5e5e5] rounded-2xl overflow-hidden"
              >
                {/* Text side */}
                <div className="p-6 sm:p-8 md:p-10">
                  <div className="w-10 h-10 rounded-xl bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center mb-4 sm:mb-5">
                    <activeFeature.icon className="w-5 h-5 text-[color:var(--fg)]" />
                  </div>
                  <h3 className="text-[18px] sm:text-[22px] font-bold mb-3 text-[color:var(--fg)]">
                    {activeFeature.title}
                  </h3>
                  <p className="text-[13.5px] sm:text-[14.5px] text-[#737373] leading-relaxed mb-5 sm:mb-6">{activeFeature.body}</p>
                  <Link href="/login" className="btn-black px-5 h-9 rounded-lg text-[13px] gap-1.5">
                    Try it free <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                </div>

                {/* Preview side */}
                <div className="border-t md:border-t-0 md:border-l border-[#e5e5e5] bg-[#f9f9f9] min-h-[200px] md:min-h-[240px]">
                  {activeFeature.preview}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </section>

        {/* ── TESTIMONIAL ── */}
        <section className="py-14 sm:py-24 px-4 sm:px-6" style={{ backgroundColor: "var(--fg)" }}>
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.65, ease: "easeOut" }}
            className="page-container text-center"
          >
            <span className="section-label block mb-8 text-[color:var(--bg)] opacity-30">Customer story</span>
            <blockquote
              className="text-2xl sm:text-3xl md:text-[42px] font-bold text-[color:var(--bg)] leading-[1.18] mb-8 max-w-3xl mx-auto"
              style={{ letterSpacing: "-0.03em" }}
            >
              "We cut our weekly reporting time by 80%. danta-search just gets our data."
            </blockquote>
            <cite className="text-[13px] text-[color:var(--bg)] opacity-35 not-italic tracking-wide">
              — Chief Financial Officer · Manufacturing Enterprise · 2,400 employees
            </cite>
          </motion.div>
        </section>

        {/* ── FINAL CTA ── */}
        <section className="page-section text-center border-b-0">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="page-container"
          >
            <span className="section-label block mb-5">Get started</span>
            <h2 className="display-lg mb-5">
              Ready to unlock<br />your data?
            </h2>
            <p className="body-lead mb-10 max-w-sm mx-auto">
              Join teams who've replaced hours of manual reporting with a single question.
            </p>
            <div className="flex gap-3 justify-center flex-wrap">
              <Link href="/login" className="btn-black px-8 h-12 rounded-xl gap-2 text-[14px]">
                Get Started Free <ArrowRight className="w-4 h-4" />
              </Link>
              <Link href="#" className="btn-outline px-8 h-12 rounded-xl text-[14px]">
                Talk to Sales
              </Link>
            </div>
          </motion.div>
        </section>
      </main>

      {/* ── FOOTER ── */}
      <footer className="border-t border-[#e5e5e5] px-4 sm:px-6 py-5">
        <div className="page-container flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="text-[13.5px] font-semibold tracking-tight">
            danta-search
          </span>
          <nav className="flex flex-wrap justify-center gap-4 sm:gap-6">
            {["Product", "Privacy", "Terms", "Contact"].map((l) => (
              <Link key={l} href="#" className="nav-link text-[12px]">{l}</Link>
            ))}
          </nav>
          <span className="text-[12px] text-[color:var(--fg-subtle)]">© 2026 danta-search</span>
        </div>
      </footer>
    </div>
  );
}
