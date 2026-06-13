"use client";

import Link from "next/link";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

/* Shared marketing nav — extracted from the landing page so every public page
   (Product, Pricing, About, Contact, Privacy, Terms) carries identical chrome.
   "Solutions" is an in-page anchor to the Capabilities block on the home page. */
const NAV_LINKS: { label: string; href: string }[] = [
  { label: "Product", href: "/product" },
  { label: "Solutions", href: "/#solutions" },
  { label: "Pricing", href: "/pricing" },
  { label: "About", href: "/about" },
];

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
        {NAV_LINKS.map((link, i) => (
          <motion.div
            key={link.label}
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.22 + i * 0.055, duration: 0.3 }}
            className="relative"
            onHoverStart={() => setHovered(link.label)}
            onHoverEnd={() => setHovered(null)}
          >
            <AnimatePresence>
              {hovered === link.label && (
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
              href={link.href}
              className="relative z-10 flex items-center px-3.5 py-1.5 text-[13px] font-medium text-white/70 hover:text-white transition-colors duration-150 rounded-full select-none"
            >
              {link.label}
            </Link>
          </motion.div>
        ))}
      </motion.div>
    </nav>
  );
}

export default function SiteNav() {
  return (
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
          <span className="text-[14px] font-semibold tracking-tight">danta-search</span>
        </Link>

        {/* Center black pill nav */}
        <NavPill />

        {/* Right actions */}
        <div className="flex items-center gap-2 sm:gap-3">
          <Link href="/login" className="nav-link hidden sm:inline text-[13px]">Sign in</Link>
          <Link href="/contact" className="btn-black px-3 sm:px-4 h-8 rounded-lg text-[12.5px] sm:text-[13px]">
            Book a Demo
          </Link>
        </div>
      </div>
    </motion.header>
  );
}
