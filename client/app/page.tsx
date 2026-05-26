import Link from "next/link";
import { Network, Search, FileStack, BarChart3, Zap, ShieldCheck } from "lucide-react";

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Navbar */}
      <header className="sticky top-0 z-50 bg-surface/80 backdrop-blur-sm border-b border-border">
        <div className="mx-auto max-w-5xl px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-bold text-foreground text-base tracking-tight">
              danta<span className="text-primary">-search</span>
            </span>
            <span className="hidden sm:inline-block px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-medium uppercase tracking-wide">
              Beta
            </span>
          </div>
          <Link
            href="/login"
            className="inline-flex items-center justify-center h-8 px-4 rounded-md bg-primary text-primary-foreground text-sm font-medium transition-opacity hover:opacity-90"
          >
            Sign In
          </Link>
        </div>
      </header>

      {/* Hero */}
      <main className="flex-1">
        <section className="flex flex-col items-center justify-center text-center px-4 pt-24 pb-16">
          <span className="inline-block mb-5 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium">
            Enterprise Data Intelligence
          </span>
          <h1 className="text-4xl md:text-6xl font-bold text-foreground leading-tight max-w-3xl">
            Chat with your
            <br />
            Excel & PDF data —{" "}
            <span className="text-primary">at scale</span>
          </h1>
          <p className="mt-5 text-base md:text-lg text-muted-foreground max-w-lg">
            Ask business questions across GBs of spreadsheets and documents in plain English.
            danta-search understands your data, your columns, and your business logic.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 mt-8">
            <Link
              href="/login"
              className="inline-flex items-center justify-center h-10 px-6 rounded-md bg-primary text-primary-foreground text-sm font-medium transition-opacity hover:opacity-90"
            >
              Get Started Free
            </Link>
            <Link
              href="/demo"
              className="inline-flex items-center justify-center h-10 px-6 rounded-md border border-border text-foreground text-sm font-medium hover:bg-muted transition-colors"
            >
              See a Demo
            </Link>
          </div>

          {/* Social proof pill */}
          <p className="mt-6 text-xs text-muted-foreground">
            Handles files up to{" "}
            <span className="text-foreground font-medium">10 GB+</span> · Excel, CSV, PDF, DOCX
          </p>
        </section>

        {/* How it works — simple 3-step flow */}
        <section className="mx-auto max-w-5xl px-4 pb-20">
          <div className="flex flex-col md:flex-row items-start md:items-center gap-2 justify-center mb-10 text-xs text-muted-foreground">
            {["Upload your files", "danta-search indexes them", "Ask anything in plain English"].map(
              (step, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-primary/10 text-primary font-semibold text-[10px]">
                    {i + 1}
                  </span>
                  <span>{step}</span>
                  {i < 2 && (
                    <span className="hidden md:inline text-border-strong">→</span>
                  )}
                </div>
              )
            )}
          </div>

          {/* Mock chat preview */}
          <div className="bg-surface border border-border rounded-xl p-5 max-w-2xl mx-auto mb-16 text-left shadow-sm">
            <p className="text-xs text-muted-foreground mb-4 font-medium uppercase tracking-wide">
              Example conversation
            </p>
            <div className="space-y-3">
              <div className="flex justify-end">
                <div className="bg-primary/10 text-foreground text-sm px-4 py-2 rounded-xl rounded-tr-sm max-w-xs">
                  What were the top 5 SKUs by revenue last quarter across all regions?
                </div>
              </div>
              <div className="flex justify-start">
                <div className="bg-muted text-foreground text-sm px-4 py-2 rounded-xl rounded-tl-sm max-w-sm">
                  Based on your Q3 sales data (3 Excel files, 142k rows):
                  <br />
                  <span className="font-medium">1. SKU-4821</span> — ₹28.4L &nbsp;
                  <span className="font-medium">2. SKU-0093</span> — ₹21.1L &nbsp;
                  <span className="text-muted-foreground text-xs">…and 3 more</span>
                </div>
              </div>
              <div className="flex justify-end">
                <div className="bg-primary/10 text-foreground text-sm px-4 py-2 rounded-xl rounded-tr-sm max-w-xs">
                  Show me SKU-4821's month-over-month trend as a table.
                </div>
              </div>
              <div className="flex justify-start">
                <div className="bg-muted text-foreground text-sm px-4 py-2 rounded-xl rounded-tl-sm max-w-sm">
                  Pulling from <span className="text-primary font-medium">sales_q3_south.xlsx</span> and <span className="text-primary font-medium">sales_q3_north.xlsx</span>…
                  <br />
                  <span className="text-muted-foreground text-xs">✓ Results ready</span>
                </div>
              </div>
            </div>
          </div>

          {/* Feature cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
            <div className="bg-surface border border-border rounded-lg p-5 text-left">
              <BarChart3 className="w-5 h-5 text-primary mb-3" />
              <h3 className="text-sm font-semibold text-foreground mb-1">
                Business-aware analysis
              </h3>
              <p className="text-xs text-muted-foreground">
                Understands column semantics, date ranges, and business metrics — not just raw cell values.
              </p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-5 text-left">
              <Zap className="w-5 h-5 text-primary mb-3" />
              <h3 className="text-sm font-semibold text-foreground mb-1">
                GB-scale performance
              </h3>
              <p className="text-xs text-muted-foreground">
                Processes and queries across millions of rows from multiple Excel, CSV, and PDF files without slowdown.
              </p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-5 text-left">
              <ShieldCheck className="w-5 h-5 text-primary mb-3" />
              <h3 className="text-sm font-semibold text-foreground mb-1">
                Source-cited answers
              </h3>
              <p className="text-xs text-muted-foreground">
                Every answer links back to the exact file, sheet, and rows it came from — fully auditable.
              </p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-5 text-left">
              <Network className="w-5 h-5 text-primary mb-3" />
              <h3 className="text-sm font-semibold text-foreground mb-1">
                Graph RAG engine
              </h3>
              <p className="text-xs text-muted-foreground">
                Builds a knowledge graph across your files to answer cross-document questions with linked reasoning.
              </p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-5 text-left">
              <Search className="w-5 h-5 text-primary mb-3" />
              <h3 className="text-sm font-semibold text-foreground mb-1">
                Hybrid search
              </h3>
              <p className="text-xs text-muted-foreground">
                Combines vector similarity and keyword search for accurate results on both structured and unstructured content.
              </p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-5 text-left">
              <FileStack className="w-5 h-5 text-primary mb-3" />
              <h3 className="text-sm font-semibold text-foreground mb-1">
                Multi-format ingestion
              </h3>
              <p className="text-xs text-muted-foreground">
                Handles Excel (XLSX), CSV, PDF, DOCX, and TXT — upload once, query across all formats together.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="text-center text-xs text-subtle-foreground pb-8">
        danta-search &copy; 2025 · Enterprise Data Intelligence
      </footer>
    </div>
  );
}