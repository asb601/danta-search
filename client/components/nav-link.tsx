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

export function IslandNavLink({ href, icon: Icon, label }: NavLinkProps) {
  const pathname = usePathname();
  const isActive = pathname === href || pathname.startsWith(href + "/");

  return (
    <Link
      href={href}
      className={cn(
        "relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors duration-150 whitespace-nowrap select-none",
        isActive
          ? "text-[color:var(--fg)]"
          : "text-[color:var(--fg-muted)] hover:text-[color:var(--fg)] hover:bg-[color:var(--surface)]"
      )}
    >
      {isActive && (
        <motion.span
          layoutId="nav-active-pill"
          className="absolute inset-0 rounded-lg"
          style={{
            backgroundColor: "var(--surface-raised)",
          }}
          transition={{ type: "spring", stiffness: 400, damping: 32 }}
        />
      )}
      <Icon className="w-3.5 h-3.5 shrink-0 relative z-10" />
      <span className="relative z-10">{label}</span>
    </Link>
  );
}

export function MobileNavLink({ href, icon: Icon, label }: NavLinkProps) {
  const pathname = usePathname();
  const isActive = pathname === href || pathname.startsWith(href + "/");

  return (
    <Link
      href={href}
      className={cn(
        "flex flex-col items-center justify-center gap-1 flex-1 py-2.5 text-xs transition-colors relative",
        isActive
          ? "text-[color:var(--fg)]"
          : "text-[color:var(--fg-subtle)] hover:text-[color:var(--fg)]"
      )}
    >
      {isActive && (
        <motion.span
          layoutId="mobile-nav-indicator"
          className="absolute top-0 left-1/2 -translate-x-1/2 w-6 h-0.5 rounded-full"
          style={{ backgroundColor: "var(--fg)" }}
          transition={{ type: "spring", stiffness: 420, damping: 34 }}
        />
      )}
      <Icon className="w-5 h-5" />
      <span className={cn("text-[10px]", isActive ? "font-semibold" : "font-normal")}>{label}</span>
    </Link>
  );
}

export function NavLink({ href, icon: Icon, label }: NavLinkProps) {
  return <IslandNavLink href={href} icon={Icon} label={label} />;
}
