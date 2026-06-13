import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, BarChart3, Zap, Search, ShieldCheck, Database, GitBranch } from "lucide-react";
import SiteNav from "@/components/marketing/SiteNav";
import SiteFooter from "@/components/marketing/SiteFooter";

export const metadata: Metadata = {
  title: "Product — danta-search",
  description:
    "Upload your business files, ask questions in plain English, and get grounded, source-cited answers across gigabytes of data.",
};

const CAPABILITIES = [
  {
    icon: BarChart3,
    title: "Understands your business logic",
    body: "Column semantics, KPIs, date ranges, and fiscal calendars are learned at ingestion — not guessed at query time.",
  },
  {
    icon: Zap,
    title: "GB-scale, instant answers",
    body: "Apache DataFusion scans Parquet with predicate pushdown across your whole data estate. No sampling, no slowdowns.",
  },
  {
    icon: Search,
    title: "Hybrid retrieval",
    body: "OpenSearch combines keyword (BM25) and vector search with native RRF ranking across structured tables and documents.",
  },
  {
    icon: ShieldCheck,
    title: "Every answer is auditable",
    body: "Results link back to the exact file, sheet, and row — the paper trail finance and compliance teams need.",
  },
  {
    icon: Database,
    title: "Any file, any size",
    body: "Excel, CSV, Parquet, PDF, and DOCX up to 10 GB+. Files are cleaned, typed, and indexed automatically.",
  },
  {
    icon: GitBranch,
    title: "Cross-file relationships",
    body: "The platform detects how your datasets relate, so questions can span multiple files without manual joins.",
  },
];

const PIPELINE = [
  { n: "01", title: "Retrieve", body: "Hybrid search surfaces the files and tables relevant to your question." },
  { n: "02", title: "Plan", body: "A semantic planner resolves entities and selects validated join paths." },
  { n: "03", title: "Execute", body: "DataFusion runs deterministic analytical SQL over Parquet — lazily, at scale." },
  { n: "04", title: "Answer", body: "A grounded response is synthesized with evidence linked to source rows." },
];

export default function ProductPage() {
  return (
    <div className="flex flex-col min-h-screen bg-white text-[color:var(--fg)]">
      <SiteNav />

      <main className="flex-1">
        {/* ── HERO ── */}
        <section className="relative overflow-hidden pt-16 pb-10 sm:pt-24 sm:pb-12 px-4 sm:px-6">
          <div
            className="absolute inset-0 grid-bg pointer-events-none"
            style={{
              opacity: 0.4,
              maskImage: "radial-gradient(ellipse 80% 55% at 50% 0%, black 30%, transparent 100%)",
            }}
          />
          <div className="relative page-container text-center">
            <span className="section-label mb-5 inline-block">The product</span>
            <h1 className="display-lg mb-5 max-w-3xl mx-auto">
              An analytics platform that <span className="gradient-text">answers, not just searches.</span>
            </h1>
            <p className="body-lead max-w-[560px] mx-auto mb-9">
              Upload your files, ask a question, and get a grounded answer with the exact rows it came
              from. No SQL, no dashboards to configure, no data engineering.
            </p>
            <div className="flex gap-3 justify-center flex-wrap">
              <Link href="/login" className="btn-black px-6 h-11 rounded-xl gap-2">
                Get Started Free <ArrowRight className="w-4 h-4" />
              </Link>
              <Link href="/pricing" className="btn-outline px-6 h-11 rounded-xl">
                See Pricing
              </Link>
            </div>
          </div>
        </section>

        {/* ── CAPABILITIES GRID ── */}
        <section className="page-section">
          <div className="page-container">
            <span className="section-label block mb-4">Capabilities</span>
            <h2 className="display-md mb-10 max-w-lg">Built for enterprise-grade data at real scale.</h2>
            <div className="border border-[#e5e5e5] rounded-2xl overflow-hidden grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
              {CAPABILITIES.map((c) => (
                <div key={c.title} className="feature-card">
                  <div className="w-10 h-10 rounded-xl bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center mb-4">
                    <c.icon className="w-5 h-5 text-[color:var(--fg)]" />
                  </div>
                  <h3 className="text-[15px] font-semibold mb-2 text-[color:var(--fg)]">{c.title}</h3>
                  <p className="text-[13.5px] text-[#737373] leading-relaxed">{c.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── PIPELINE ── */}
        <section className="page-section">
          <div className="page-container">
            <span className="section-label block mb-4">How a question is answered</span>
            <h2 className="display-md mb-10 max-w-lg">Four steps, fully deterministic.</h2>
            <div className="border border-[#e5e5e5] rounded-2xl overflow-hidden grid grid-cols-1 md:grid-cols-4">
              {PIPELINE.map((s) => (
                <div key={s.n} className="step-card">
                  <div className="step-number">{s.n}</div>
                  <h3 className="text-[15px] font-semibold mb-2 text-[color:var(--fg)]">{s.title}</h3>
                  <p className="text-[13.5px] text-[#737373] leading-relaxed">{s.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section className="page-section text-center border-b-0">
          <div className="page-container">
            <span className="section-label block mb-5">Get started</span>
            <h2 className="display-lg mb-5">Put it on your own data.</h2>
            <p className="body-lead mb-10 max-w-sm mx-auto">
              Replace hours of manual reporting with a single question.
            </p>
            <div className="flex gap-3 justify-center flex-wrap">
              <Link href="/login" className="btn-black px-8 h-12 rounded-xl gap-2 text-[14px]">
                Get Started Free <ArrowRight className="w-4 h-4" />
              </Link>
              <Link href="/contact" className="btn-outline px-8 h-12 rounded-xl text-[14px]">
                Talk to Sales
              </Link>
            </div>
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
}
