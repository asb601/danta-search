import Link from "next/link";

/* Shared marketing footer — links resolve to the real public routes. */
const FOOTER_LINKS: { label: string; href: string }[] = [
  { label: "Product", href: "/product" },
  { label: "Pricing", href: "/pricing" },
  { label: "About", href: "/about" },
  { label: "Contact", href: "/contact" },
  { label: "Privacy", href: "/privacy" },
  { label: "Terms", href: "/terms" },
];

export default function SiteFooter() {
  return (
    <footer className="border-t border-[#e5e5e5] px-4 sm:px-6 py-5">
      <div className="page-container flex flex-col sm:flex-row items-center justify-between gap-4">
        <span className="text-[13.5px] font-semibold tracking-tight">danta-search</span>
        <nav className="flex flex-wrap justify-center gap-4 sm:gap-6">
          {FOOTER_LINKS.map((l) => (
            <Link key={l.label} href={l.href} className="nav-link text-[12px]">{l.label}</Link>
          ))}
        </nav>
        <span className="text-[12px] text-[color:var(--fg-subtle)]">© 2026 danta-search</span>
      </div>
    </footer>
  );
}
