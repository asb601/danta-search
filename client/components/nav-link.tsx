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

/** Horizontal top-bar nav link — used in the app shell top nav */
export function NavLink({ href, icon: Icon, label }: NavLinkProps) {
  const pathname = usePathname();
  const isActive = pathname === href || pathname.startsWith(href + "/");

  return (
    <Link
      href={href}
      className={cn(
        "relative flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[13px] font-medium transition-colors duration-150 whitespace-nowrap",
        isActive ? "text-white" : "text-[#737373] hover:text-[#0a0a0a] hover:bg-[#f4f4f4]"
      )}
    >
      {isActive && (
        <motion.span
          layoutId="topnav-active-pill"
          className="absolute inset-0 rounded-full bg-[#0a0a0a]"
          style={{ boxShadow: "inset 0 1px 0 rgba(255,255,255,0.12), 0 1px 4px rgba(0,0,0,0.12)" }}
          transition={{ type: "spring", stiffness: 420, damping: 36 }}
        />
      )}
      <Icon className={cn("w-3.5 h-3.5 shrink-0 relative z-10", isActive ? "text-white" : "text-[#a3a3a3]")} />
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
        isActive ? "text-[#0a0a0a]" : "text-[#a3a3a3] hover:text-[#0a0a0a]"
      )}
    >
      {isActive && (
        <motion.span
          layoutId="mobile-nav-indicator"
          className="absolute top-0 left-1/2 -translate-x-1/2 w-6 h-0.5 rounded-full bg-[#0a0a0a]"
        />
      )}
      <Icon className="w-5 h-5" />
      <span className={cn("text-[10px]", isActive ? "font-semibold" : "font-normal")}>{label}</span>
    </Link>
  );
}
