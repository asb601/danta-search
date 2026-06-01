"use client";

import { useRouter, usePathname } from "next/navigation";
import { useEffect, useCallback, useState } from "react";
import {
  MessageSquare,
  FolderOpen,
  UserCircle,
  LayoutDashboard,
} from "lucide-react";
import { motion } from "framer-motion";
import { IslandNavLink, MobileNavLink } from "@/components/nav-link";
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
          // A 403 onboarding gate ({ onboarding_required: true }) means the
          // owner must finish the wizard — block and redirect to setup.
          if (res.status === 403) {
            try {
              const body = await res.clone().json();
              if (body?.onboarding_required === true) {
                if (!cancelled) setOwnerOnboarding("blocked");
                return;
              }
            } catch {
              /* not JSON — fall through */
            }
          }
          // Otherwise can't determine → don't trap the owner; normal routing.
          if (!cancelled) setOwnerOnboarding("ok");
          return;
        }
        const data = (await res.json()) as {
          state?: string;
          completed?: boolean;
          onboarding_required?: boolean;
        };
        if (cancelled) return;
        // Completed via either the legacy `state === "completed"` signal or an
        // explicit `completed` flag. Anything else (not-completed) → blocked.
        const done = data.state === "completed" || data.completed === true;
        setOwnerOnboarding(done ? "ok" : "blocked");
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

  // Main nav — Settings consolidates Containers / Logs / Profile / Org
  const mainNavItems: NavItem[] = [
    { href: "/chat",       icon: MessageSquare,   label: "Chat"       },
    { href: "/folders",    icon: FolderOpen,      label: "Folders"    },
    { href: "/dashboards", icon: LayoutDashboard, label: "Dashboards" },
    { href: "/settings",   icon: UserCircle,      label: "Settings"   },
  ];

  // Mobile keeps same 4 items
  const mobileNavItems = mainNavItems;

  return (
    <div className="flex flex-col h-screen bg-white overflow-hidden">

      {/* ── Dynamic Island (desktop only, fixed floating) ─────────── */}
      <motion.div
        initial={{ opacity: 0, scale: 0.82, y: -12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 22, delay: 0.05 }}
        className="hidden md:flex fixed top-3 left-1/2 -translate-x-1/2 z-50 items-center gap-0.5 rounded-full px-1.5 py-1.5"
        style={{
          background: "linear-gradient(180deg, #2c2b28 0%, #1a1918 100%)",
          boxShadow: "0 4px 32px rgba(0,0,0,0.22), 0 0 0 1px rgba(255,255,255,0.05), inset 0 1px 0 rgba(255,255,255,0.08)",
        }}
      >
        {/* Brand mark */}
        <div className="flex items-center gap-1.5 pl-1.5 pr-2 mr-0.5">
          <div className="w-5 h-5 rounded-full bg-white/12 border border-white/10 flex items-center justify-center">
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
              <circle cx="6" cy="6" r="3.5" stroke="white" strokeWidth="1.5"/>
              <circle cx="6" cy="6" r="1.25" fill="white"/>
            </svg>
          </div>
          <span
            className="text-[13px] font-bold text-white/80 tracking-tight"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}
          >
            danta
          </span>
        </div>

        {/* Separator */}
        <div className="w-px h-4 bg-white/10 mx-1 shrink-0" />

        {/* Nav items */}
        {mainNavItems.map((item, i) => (
          <motion.div
            key={item.href}
            initial={{ opacity: 0, y: -3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12 + i * 0.05, duration: 0.2 }}
          >
            <IslandNavLink {...item} />
          </motion.div>
        ))}
      </motion.div>

      {/* Spacer so content doesn't hide under the island */}
      <div className="hidden md:block h-[58px] shrink-0" />

      {/* ── Page content ─────────────────────────────────────────── */}
      <main className="flex-1 overflow-hidden pb-14 md:pb-0">
        {children}
      </main>

      {/* ── Mobile bottom nav ────────────────────────────────────── */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-[#e5e5e5] flex items-stretch z-50">
        {mobileNavItems.map((item) => (
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
