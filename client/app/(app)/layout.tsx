"use client";

import { useRouter, usePathname } from "next/navigation";
import { useEffect, useCallback, useState } from "react";
import {
  MessageSquare,
  FolderOpen,
  LogOut,
  Database,
  UserCircle,
  ScrollText,
  LayoutDashboard,
} from "lucide-react";
import { motion } from "framer-motion";
import { Building2 } from "lucide-react";
import { NavLink, MobileNavLink } from "@/components/nav-link";
import { AuthProvider, useAuth } from "@/components/auth-provider";
import { useIdleTimeout } from "@/hooks/use-idle-timeout";
import { capabilitiesFor, getRole } from "@/lib/roles";
import { apiFetch } from "@/lib/auth";

const IDLE_TIMEOUT_MS = 30 * 60 * 1000;

interface NavItem {
  href: string;
  icon: typeof MessageSquare;
  label: string;
}

function AppShellInner({ children }: { children: React.ReactNode }) {
  const { user, loading, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const noNavRoutes = ["/onboarding"];
  const hideNav = noNavRoutes.some(
    (r) => pathname === r || pathname.startsWith(r + "/"),
  );
  const onOnboarding =
    pathname === "/onboarding" || pathname.startsWith("/onboarding/");

  // HARD ONBOARDING GATE (frontend). For an org_owner whose org has NOT
  // finished onboarding, force the user into the wizard and prevent any other
  // app page from rendering. "completed" === done. Other roles are unaffected.
  // States: "unknown" (still checking), "blocked" (force wizard), "ok" (pass).
  const isOrgOwner = !!user && getRole(user) === "org_owner";
  const [ownerOnboarding, setOwnerOnboarding] = useState<
    "unknown" | "blocked" | "ok"
  >("unknown");

  useEffect(() => {
    if (loading || !user) return;
    if (!isOrgOwner) {
      setOwnerOnboarding("ok"); // non-owners are never gated
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch("/api/onboarding/state");
        if (!res.ok) {
          // Can't determine → don't trap the owner; let normal routing apply.
          if (!cancelled) setOwnerOnboarding("ok");
          return;
        }
        const data = (await res.json()) as { state?: string };
        if (cancelled) return;
        setOwnerOnboarding(data.state === "completed" ? "ok" : "blocked");
      } catch {
        if (!cancelled) setOwnerOnboarding("ok");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loading, user, isOrgOwner]);

  useEffect(() => {
    if (ownerOnboarding === "blocked" && !onOnboarding) {
      router.replace("/onboarding/setup");
    }
  }, [ownerOnboarding, onOnboarding, router]);

  const handleIdle = useCallback(() => {
    logout();
    router.replace("/login");
  }, [logout, router]);

  useIdleTimeout({ timeoutMs: IDLE_TIMEOUT_MS, onTimeout: handleIdle });

  useEffect(() => {
    if (loading) return;
    if (!user) {
      document.cookie = "token=; path=/; max-age=0";
      router.replace("/login");
      return;
    }
    if (
      !user.is_admin &&
      !user.allowed_domains &&
      !(pathname === "/onboarding" || pathname.startsWith("/onboarding/"))
    ) {
      router.replace("/onboarding");
    }
  }, [loading, user, router, pathname]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-white">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-[#0a0a0a]/8 flex items-center justify-center">
            <div className="w-4 h-4 rounded-full border-2 border-[#0a0a0a] border-t-transparent animate-spin" />
          </div>
          <p className="text-[13px] text-[#a3a3a3]">Loading…</p>
        </div>
      </div>
    );
  }

  if (!user) return null;

  // Org owner whose onboarding state is still resolving, or who is blocked and
  // not yet on the wizard route — render nothing (the redirect effect above is
  // sending them to /onboarding/setup). Prevents any app page from flashing.
  if (isOrgOwner && !onOnboarding && ownerOnboarding !== "ok") {
    return (
      <div className="flex h-screen items-center justify-center bg-white">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-[#0a0a0a]/8 flex items-center justify-center">
            <div className="w-4 h-4 rounded-full border-2 border-[#0a0a0a] border-t-transparent animate-spin" />
          </div>
          <p className="text-[13px] text-[#a3a3a3]">Loading…</p>
        </div>
      </div>
    );
  }

  if (hideNav) {
    return <div className="h-screen bg-white">{children}</div>;
  }

  const caps = capabilitiesFor(user);

  const navItems: NavItem[] = [
    { href: "/chat",            icon: MessageSquare,   label: "Chat"       },
    { href: "/folders",         icon: FolderOpen,      label: "Folders"    },
    { href: "/dashboards",      icon: LayoutDashboard, label: "Dashboards" },
    ...(caps.canManageOrganizations
      ? [{ href: "/admin/organizations", icon: Building2, label: "Organizations" }]
      : []),
    ...(user.is_admin || user.role === "developer"
      ? [{ href: "/admin/containers", icon: Database,   label: "Containers" }]
      : []),
    ...(user.is_admin || user.role === "developer" || user.role === "manager" || user.role === "user"
      ? [{ href: "/admin/logs",       icon: ScrollText, label: "Logs"       }]
      : []),
    { href: "/profile",         icon: UserCircle,      label: "Profile"    },
  ];

  const roleLabel = user.is_admin
    ? { text: "Admin",     cls: "bg-[#0a0a0a]/8 text-[#0a0a0a]" }
    : user.role === "developer"
    ? { text: "Developer", cls: "bg-violet-500/10 text-violet-600" }
    : user.role === "manager"
    ? { text: "Manager",   cls: "bg-cyan-500/10 text-cyan-700" }
    : null;

  return (
    <div className="flex flex-col h-screen bg-white overflow-hidden">

      {/* ── Desktop top nav bar ──────────────────────────────────── */}
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="hidden md:flex items-center gap-0 h-[52px] shrink-0 border-b border-[#e5e5e5] bg-white px-4 relative z-30"
      >
        {/* Brand */}
        <div className="flex items-center gap-2 mr-5 shrink-0">
          <div
            className="w-6 h-6 rounded-lg bg-[#0a0a0a] flex items-center justify-center"
            style={{ boxShadow: "0 1px 4px rgba(0,0,0,0.18)" }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <circle cx="6" cy="6" r="3.5" stroke="white" strokeWidth="1.5"/>
              <circle cx="6" cy="6" r="1.25" fill="white"/>
            </svg>
          </div>
          <span
            className="text-[14px] font-bold text-[#0a0a0a] tracking-tight"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.025em" }}
          >
            danta
          </span>
        </div>

        {/* Nav divider */}
        <div className="w-px h-4 bg-[#e5e5e5] mr-3 shrink-0" />

        {/* Nav items */}
        <nav className="flex items-center gap-0.5 flex-1 min-w-0 overflow-x-auto" style={{ scrollbarWidth: "none" }}>
          {navItems.map((item, i) => (
            <motion.div
              key={item.href}
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.06 + i * 0.04, duration: 0.22, ease: "easeOut" }}
            >
              <NavLink {...item} />
            </motion.div>
          ))}
        </nav>

        {/* Right side — user + sign out */}
        <div className="flex items-center gap-2 ml-3 shrink-0">
          {roleLabel && (
            <motion.span
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, duration: 0.2 }}
              className={`text-[10px] font-semibold tracking-wide uppercase px-2 py-0.5 rounded-full ${roleLabel.cls}`}
            >
              {roleLabel.text}
            </motion.span>
          )}

          <div className="w-px h-4 bg-[#e5e5e5]" />

          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.94 }}
            onClick={logout}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[12px] text-[#a3a3a3] hover:text-[#dc2626] hover:bg-[#dc2626]/6 transition-colors"
            title="Sign out"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span className="hidden lg:inline">Sign out</span>
          </motion.button>
        </div>
      </motion.header>

      {/* ── Page content ─────────────────────────────────────────── */}
      <main className="flex-1 overflow-hidden pb-14 md:pb-0">
        {children}
      </main>

      {/* ── Mobile bottom nav (unchanged) ────────────────────────── */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-[#e5e5e5] flex items-stretch z-50">
        {navItems.map((item) => (
          <MobileNavLink key={item.href} {...item} />
        ))}
      </nav>
    </div>
  );
}

export default function AppShellLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <AppShellInner>{children}</AppShellInner>
    </AuthProvider>
  );
}
