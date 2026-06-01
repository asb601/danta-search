"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

interface NavLinkProps {
  href: string;
  icon: LucideIcon;
  label: string;
}

/** Inside the Dynamic Island — dark background, white text */
export function IslandNavLink({ href, icon: Icon, label }: NavLinkProps) {
  const pathname = usePathname();
  const isActive = pathname === href || pathname.startsWith(href + "/");

  return (
    <Link
      href={href}
      className={cn(
        "relative flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-[13px] font-medium transition-colors duration-150 whitespace-nowrap select-none",
        isActive ? "" : "text-white/60 hover:text-white/90"
      )}
      style={isActive ? { color: "var(--fg)" } : undefined}
    >
      {isActive && (
        <motion.span
          layoutId="island-active-pill"
          className="absolute inset-0 rounded-full"
          style={{ backgroundColor: "var(--bg)", boxShadow: "0 1px 6px rgba(0,0,0,0.15)" }}
          transition={{ type: "spring", stiffness: 420, damping: 34 }}
        />
      )}
      <Icon className={cn("w-3.5 h-3.5 shrink-0 relative z-10", isActive ? "text-[#1a1918]" : "text-white/50")}
        style={isActive ? { color: "var(--fg)" } : undefined}
      />
      <span className="relative z-10">{label}</span>
    </Link>
  );
}

/** Mobile bottom nav link */
export function MobileNavLink({ href, icon: Icon, label }: NavLinkProps) {
  const pathname = usePathname();
  const isActive = pathname === href || pathname.startsWith(href + "/");

  return (
    <Link
      href={href}
      className={cn(
        "flex flex-col items-center justify-center gap-1 flex-1 py-2.5 text-xs transition-colors relative",
        isActive ? "text-[#0a0a0a]" : "text-[#a3a3a3] hover:text-[#0a0a0a]"
      )}
    >
      {isActive && (
        <motion.span
          layoutId="mobile-nav-indicator"
          className="absolute top-0 left-1/2 -translate-x-1/2 w-6 h-0.5 rounded-full bg-[#0a0a0a]"
          transition={{ type: "spring", stiffness: 420, damping: 34 }}
        />
      )}
      <Icon className="w-5 h-5" />
      <span className={cn("text-[10px]", isActive ? "font-semibold" : "font-normal")}>{label}</span>
    </Link>
  );
}

/** Legacy — kept for any remaining usage */
export function NavLink({ href, icon: Icon, label }: NavLinkProps) {
  return <IslandNavLink href={href} icon={Icon} label={label} />;
}
