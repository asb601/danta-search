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

export function NavLink({ href, icon: Icon, label }: NavLinkProps) {
  const pathname = usePathname();
  const isActive = pathname === href || pathname.startsWith(href + "/");

  return (
    <Link
      href={href}
      className={cn(
        "sidebar-item relative",
        isActive ? "active text-[#0a0a0a]" : "text-[#737373]"
      )}
    >
      {/* Sliding background pill */}
      {isActive && (
        <motion.span
          layoutId="sidebar-active-pill"
          className="absolute inset-0 bg-[#f0f0f0] rounded-[6px]"
          transition={{ type: "spring", stiffness: 420, damping: 36 }}
        />
      )}
      <Icon className={cn("w-4 h-4 shrink-0 relative z-10", isActive ? "text-[#0a0a0a]" : "text-[#a3a3a3]")} />
      <span className="relative z-10 text-[13px]">{label}</span>
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
