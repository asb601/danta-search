"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
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
        "relative flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors duration-150",
        isActive
          ? "nav-item-active"
          : "text-muted-foreground hover:text-foreground hover:bg-muted"
      )}
    >
      <Icon className={cn("w-4 h-4 shrink-0", isActive ? "text-primary" : "")} />
      <span>{label}</span>
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
        isActive ? "text-primary" : "text-muted-foreground hover:text-foreground"
      )}
    >
      {isActive && (
        <span className="absolute top-0 left-1/2 -translate-x-1/2 w-6 h-0.5 rounded-full bg-primary" />
      )}
      <Icon className="w-5 h-5" />
      <span className={cn("text-[10px]", isActive ? "font-semibold" : "font-normal")}>{label}</span>
    </Link>
  );
}
