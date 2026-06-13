import type { Metadata } from "next";
import { Mail, MapPin, Clock, ArrowRight } from "lucide-react";
import SiteNav from "@/components/marketing/SiteNav";
import SiteFooter from "@/components/marketing/SiteFooter";

export const metadata: Metadata = {
  title: "Contact — danta-search",
  description: "Get in touch with the danta-search team — sales, support, or partnerships.",
};

/* Placeholder contact details — swap in real values. */
const DETAILS = [
  { icon: Mail, label: "Email", value: "hello@dantasearch.com", href: "mailto:hello@dantasearch.com" },
  { icon: MapPin, label: "Office", value: "Hyderabad, India", href: null },
  { icon: Clock, label: "Hours", value: "Mon–Fri · 9:00–18:00 IST", href: null },
];

export default function ContactPage() {
  return (
    <div className="flex flex-col min-h-screen bg-white text-[color:var(--fg)]">
      <SiteNav />

      <main className="flex-1">
        {/* ── HERO ── */}
        <section className="relative overflow-hidden pt-16 pb-8 sm:pt-24 sm:pb-10 px-4 sm:px-6">
          <div
            className="absolute inset-0 grid-bg pointer-events-none"
            style={{
              opacity: 0.4,
              maskImage: "radial-gradient(ellipse 80% 55% at 50% 0%, black 30%, transparent 100%)",
            }}
          />
          <div className="relative page-container text-center">
            <span className="section-label mb-5 inline-block">Contact</span>
            <h1 className="display-lg mb-5 max-w-2xl mx-auto">
              Let&apos;s <span className="gradient-text">talk.</span>
            </h1>
            <p className="body-lead max-w-[480px] mx-auto">
              Questions about the product, pricing, or a demo? Send us a note and we&apos;ll get back to you.
            </p>
          </div>
        </section>

        {/* ── CONTACT GRID ── */}
        <section className="px-4 sm:px-6 pb-16 sm:pb-24 page-container mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-[1fr_1.2fr] gap-6 lg:gap-10 items-start">
            {/* Details */}
            <div className="flex flex-col gap-3">
              {DETAILS.map((d) => {
                const inner = (
                  <div className="flex items-center gap-4 p-5 rounded-2xl border border-[#e5e5e5] bg-[#f9f9f9] transition-colors hover:bg-[#f4f4f4]">
                    <span className="w-10 h-10 shrink-0 rounded-xl bg-white border border-[#e5e5e5] flex items-center justify-center">
                      <d.icon className="w-5 h-5 text-[color:var(--fg)]" />
                    </span>
                    <div>
                      <div className="section-label mb-0.5">{d.label}</div>
                      <div className="text-[14px] font-medium text-[color:var(--fg)]">{d.value}</div>
                    </div>
                  </div>
                );
                return d.href ? (
                  <a key={d.label} href={d.href} className="block no-underline">{inner}</a>
                ) : (
                  <div key={d.label}>{inner}</div>
                );
              })}
            </div>

            {/* Form (visual only — wire up to a backend later) */}
            <div className="rounded-2xl border border-[#e5e5e5] bg-white p-6 sm:p-8 shadow-sm">
              <h2 className="text-[18px] font-bold mb-5 text-[color:var(--fg)]">Send a message</h2>
              <form className="flex flex-col gap-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="section-label block mb-2">Name</label>
                    <input type="text" placeholder="Your name" className="field-input" />
                  </div>
                  <div>
                    <label className="section-label block mb-2">Work email</label>
                    <input type="email" placeholder="you@company.com" className="field-input" />
                  </div>
                </div>
                <div>
                  <label className="section-label block mb-2">Company</label>
                  <input type="text" placeholder="Company name" className="field-input" />
                </div>
                <div>
                  <label className="section-label block mb-2">Message</label>
                  <textarea rows={4} placeholder="How can we help?" className="field-textarea" />
                </div>
                <button type="button" className="btn-black h-11 rounded-xl gap-2 w-full justify-center text-[13.5px]">
                  Send message <ArrowRight className="w-4 h-4" />
                </button>
                <p className="text-[12px] text-[color:var(--fg-subtle)] text-center">
                  Prefer email? Reach us at{" "}
                  <a href="mailto:hello@dantasearch.com" className="underline hover:text-[color:var(--fg)]">
                    hello@dantasearch.com
                  </a>
                </p>
              </form>
            </div>
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
}
