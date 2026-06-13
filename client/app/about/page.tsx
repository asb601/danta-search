import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import SiteNav from "@/components/marketing/SiteNav";
import SiteFooter from "@/components/marketing/SiteFooter";

export const metadata: Metadata = {
  title: "About — danta-search",
  description:
    "danta-search is an enterprise analytics platform that lets teams ask business questions across their data in plain English — with grounded, source-cited answers.",
};

const VALUES = [
  {
    n: "01",
    title: "Correct over clever",
    body: "We would rather say \"I don't know\" than guess. Every answer is grounded in your real data and links back to the exact source rows.",
  },
  {
    n: "02",
    title: "Intelligence at ingestion",
    body: "We do the hard thinking when data arrives — schema, semantics, relationships — so answers at query time are fast, deterministic, and reproducible.",
  },
  {
    n: "03",
    title: "Enterprise by default",
    body: "Multi-tenant isolation, role-based access, and a full audit trail are built into the foundation, not bolted on later.",
  },
];

export default function AboutPage() {
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
            <span className="section-label mb-5 inline-block">About us</span>
            <h1 className="display-lg mb-5 max-w-3xl mx-auto">
              We help teams <span className="gradient-text">talk to their data.</span>
            </h1>
            <p className="body-lead max-w-[560px] mx-auto">
              danta-search turns gigabytes of spreadsheets, PDFs, and documents into answers you
              can trust — in plain English, with every number traced back to its source.
            </p>
          </div>
        </section>

        {/* ── MISSION ── */}
        <section className="page-section">
          <div className="page-container grid md:grid-cols-[200px_1fr] gap-8 md:gap-14">
            <span className="section-label">Our mission</span>
            <div className="max-w-2xl">
              <h2 className="display-md mb-5">
                Business answers shouldn&apos;t require an analyst, a query language, or a week of waiting.
              </h2>
              <p className="body-lead mb-4">
                Most enterprise data lives in thousands of files that only a handful of people know
                how to read. The questions are simple — &ldquo;which vendors are overdue?&rdquo;,
                &ldquo;how did Q2 compare to Q1?&rdquo; — but getting the answer means SQL,
                dashboards, and back-and-forth.
              </p>
              <p className="body-lead">
                We built danta-search to close that gap: a platform that understands your business
                logic at ingestion, then answers questions instantly and verifiably at scale.
              </p>
            </div>
          </div>
        </section>

        {/* ── VALUES ── */}
        <section className="page-section">
          <div className="page-container">
            <span className="section-label block mb-4">What we believe</span>
            <h2 className="display-md mb-10 max-w-lg">The principles behind the product.</h2>
            <div className="border border-[#e5e5e5] rounded-2xl overflow-hidden grid grid-cols-1 md:grid-cols-3">
              {VALUES.map((v) => (
                <div key={v.n} className="step-card">
                  <div className="step-number">{v.n}</div>
                  <h3 className="text-[15px] font-semibold mb-2 text-[color:var(--fg)]">{v.title}</h3>
                  <p className="text-[13.5px] text-[#737373] leading-relaxed">{v.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section className="page-section text-center border-b-0">
          <div className="page-container">
            <span className="section-label block mb-5">Get started</span>
            <h2 className="display-lg mb-5">Ready to talk to your data?</h2>
            <p className="body-lead mb-10 max-w-sm mx-auto">
              See danta-search on your own files — no setup, no data engineering.
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
