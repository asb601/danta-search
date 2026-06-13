import type { Metadata } from "next";
import Link from "next/link";
import { Check, ArrowRight } from "lucide-react";
import SiteNav from "@/components/marketing/SiteNav";
import SiteFooter from "@/components/marketing/SiteFooter";

export const metadata: Metadata = {
  title: "Pricing — danta-search",
  description: "Simple, transparent pricing for teams of every size.",
};

/* NOTE: prices and limits below are PLACEHOLDERS — swap in the real numbers.
   Each tier's `cta.href` points to sign-up or contact. */
const TIERS = [
  {
    name: "Starter",
    price: "₹0",
    cadence: "/ month",
    blurb: "For individuals exploring danta-search on their own data.",
    cta: { label: "Get Started Free", href: "/login" },
    highlighted: false,
    features: [
      "Up to 1 GB of data",
      "Single workspace",
      "Excel, CSV & PDF uploads",
      "Natural-language queries",
      "Source-cited answers",
      "Community support",
    ],
  },
  {
    name: "Growth",
    price: "₹—",
    cadence: "/ month",
    blurb: "For teams that need scale, collaboration, and dashboards.",
    cta: { label: "Start Free Trial", href: "/login" },
    highlighted: true,
    features: [
      "Up to 100 GB of data",
      "Unlimited team members",
      "Dashboards & saved views",
      "Hybrid retrieval at scale",
      "Role-based access control",
      "Priority email support",
    ],
  },
  {
    name: "Enterprise",
    price: "Custom",
    cadence: "",
    blurb: "For organizations with security, scale, and compliance needs.",
    cta: { label: "Talk to Sales", href: "/contact" },
    highlighted: false,
    features: [
      "Unlimited data volume",
      "Dedicated tenant isolation",
      "SSO & advanced RBAC",
      "Audit logs & data residency",
      "Custom integrations",
      "Dedicated success manager",
    ],
  },
];

export default function PricingPage() {
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
            <span className="section-label mb-5 inline-block">Pricing</span>
            <h1 className="display-lg mb-5 max-w-2xl mx-auto">
              Simple pricing that <span className="gradient-text">scales with you.</span>
            </h1>
            <p className="body-lead max-w-[480px] mx-auto">
              Start free. Upgrade when your team and your data grow. No hidden fees.
            </p>
          </div>
        </section>

        {/* ── TIERS ── */}
        <section className="px-4 sm:px-6 pb-16 sm:pb-24 page-container mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 items-start">
            {TIERS.map((t) => (
              <div
                key={t.name}
                className={`relative rounded-2xl border p-6 sm:p-7 flex flex-col ${
                  t.highlighted
                    ? "border-[#c2c1ba] bg-[#f9f9f9] shadow-sm md:-mt-3 md:mb-3"
                    : "border-[#e5e5e5] bg-white"
                }`}
              >
                {t.highlighted && (
                  <span className="badge-muted absolute -top-2.5 left-6">Most popular</span>
                )}
                <h3 className="text-[16px] font-semibold mb-1 text-[color:var(--fg)]">{t.name}</h3>
                <p className="text-[13px] text-[#737373] leading-relaxed mb-5 min-h-[40px]">{t.blurb}</p>

                <div className="flex items-end gap-1.5 mb-6">
                  <span className="text-[36px] font-extrabold tracking-tight leading-none text-[color:var(--fg)]">{t.price}</span>
                  {t.cadence && <span className="text-[13px] text-[#a3a3a3] mb-1">{t.cadence}</span>}
                </div>

                <Link
                  href={t.cta.href}
                  className={`${t.highlighted ? "btn-black" : "btn-outline"} h-11 rounded-xl gap-2 w-full justify-center text-[13.5px] mb-6`}
                >
                  {t.cta.label} <ArrowRight className="w-4 h-4" />
                </Link>

                <ul className="flex flex-col gap-2.5">
                  {t.features.map((f) => (
                    <li key={f} className="flex items-start gap-2.5">
                      <span className="w-4 h-4 mt-0.5 shrink-0 rounded-full bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center">
                        <Check className="w-2.5 h-2.5 text-[color:var(--fg)]" />
                      </span>
                      <span className="text-[13px] text-[#525252] leading-relaxed">{f}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <p className="text-center section-label mt-10">
            Prices shown are placeholders · All plans include source-cited, auditable answers
          </p>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
}
