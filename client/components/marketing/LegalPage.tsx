import SiteNav from "@/components/marketing/SiteNav";
import SiteFooter from "@/components/marketing/SiteFooter";

export type LegalSection = { heading: string; body: string[] };

/* Shared layout for Privacy / Terms — a narrow, readable legal column built
   purely on the existing design tokens (no prose dependency, no new CSS). */
export default function LegalPage({
  label,
  title,
  updated,
  intro,
  sections,
}: {
  label: string;
  title: string;
  updated: string;
  intro: string;
  sections: LegalSection[];
}) {
  return (
    <div className="flex flex-col min-h-screen bg-white text-[color:var(--fg)]">
      <SiteNav />

      <main className="flex-1">
        <section className="px-4 sm:px-6 pt-16 sm:pt-24 pb-16 sm:pb-24">
          <div className="page-container max-w-3xl">
            <span className="section-label mb-4 inline-block">{label}</span>
            <h1 className="display-md mb-3">{title}</h1>
            <p className="text-[13px] text-[color:var(--fg-subtle)] mb-8">Last updated · {updated}</p>
            <p className="body-lead mb-12">{intro}</p>

            <div className="flex flex-col gap-10">
              {sections.map((s, i) => (
                <div key={s.heading}>
                  <h2 className="text-[18px] font-bold mb-3 text-[color:var(--fg)]">
                    {i + 1}. {s.heading}
                  </h2>
                  {s.body.map((p, j) => (
                    <p key={j} className="text-[14.5px] text-[#525252] leading-[1.7] mb-3 last:mb-0">
                      {p}
                    </p>
                  ))}
                </div>
              ))}
            </div>

            <p className="text-[13px] text-[color:var(--fg-subtle)] mt-12 pt-6 border-t border-[#e5e5e5]">
              This document is a general template provided for transparency and may be updated. For
              questions, contact us at{" "}
              <a href="mailto:hello@dantasearch.com" className="underline hover:text-[color:var(--fg)]">
                hello@dantasearch.com
              </a>
              .
            </p>
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
}
