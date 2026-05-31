"use client";

import { useRouter, usePathname } from "next/navigation";
import { useState, useEffect, useCallback } from "react";
import {
  MessageSquare,
  FolderOpen,
  LogOut,
  PanelLeftClose,
  PanelLeft,
  Database,
  UserCircle,
  ScrollText,
  LayoutDashboard,
} from "lucide-react";
import { NavLink, MobileNavLink } from "@/components/nav-link";
import { AuthProvider, useAuth } from "@/components/auth-provider";
import { useIdleTimeout } from "@/hooks/use-idle-timeout";

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
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const noNavRoutes = ["/onboarding"];
  const hideNav = noNavRoutes.includes(pathname);

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
    if (!user.is_admin && !user.allowed_domains && pathname !== "/onboarding") {
      router.replace("/onboarding");
    }
  }, [loading, user, router, pathname]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          </div>
          <p className="text-sm text-muted-foreground">Loading…</p>
        </div>
      </div>
    );
  }

  if (!user) return null;

  if (hideNav) {
    return <div className="h-screen bg-background">{children}</div>;
  }

  const navItems: NavItem[] = [
    { href: "/chat", icon: MessageSquare, label: "Chat" },
    { href: "/folders", icon: FolderOpen, label: "Folders" },
    { href: "/dashboards", icon: LayoutDashboard, label: "Dashboards" },
    ...(user.is_admin || user.role === "developer"
      ? [{ href: "/admin/containers", icon: Database, label: "Containers" }]
      : []),
    ...(user.is_admin ||
    user.role === "developer" ||
    user.role === "manager" ||
    user.role === "user"
      ? [{ href: "/admin/logs", icon: ScrollText, label: "Logs" }]
      : []),
    { href: "/profile", icon: UserCircle, label: "Profile" },
  ];

  const roleLabel = user.is_admin
    ? { text: "Admin", cls: "bg-primary/10 text-primary" }
    : user.role === "developer"
    ? { text: "Developer", cls: "bg-violet-500/10 text-violet-600" }
    : user.role === "manager"
    ? { text: "Manager", cls: "bg-cyan-500/10 text-cyan-700" }
    : null;

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Collapsed sidebar toggle */}
      {!sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          className="hidden md:flex fixed top-4 left-4 z-40 items-center justify-center w-9 h-9 rounded-lg bg-card border border-border shadow-sm text-muted-foreground hover:text-primary hover:border-primary/40 transition-colors"
        >
          <PanelLeft className="w-4 h-4" />
        </button>
      )}

      {/* Desktop sidebar */}
      <aside
        className={`hidden md:flex flex-col shrink-0 bg-sidebar border-r border-sidebar-border h-screen sticky top-0 transition-[width] duration-200 overflow-hidden ${
          sidebarOpen ? "w-[228px]" : "w-0 border-r-0"
        }`}
      >
        {/* Brand header */}
        <div className="px-4 pt-4 pb-4 border-b border-[#e5e5e5]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <div className="w-6 h-6 rounded-lg bg-[#0a0a0a] flex items-center justify-center shrink-0">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <circle cx="6" cy="6" r="3.5" stroke="white" strokeWidth="1.5"/>
                  <circle cx="6" cy="6" r="1.25" fill="white"/>
                </svg>
              </div>
              <div className="min-w-0">
                <p className="text-[13px] font-semibold tracking-tight text-[#0a0a0a] leading-none mb-0.5" style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}>
                  danta-search
                </p>
                <p className="text-[11px] text-[#a3a3a3] truncate">{user.email}</p>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="btn-ghost p-1.5 rounded-md"
            >
              <PanelLeftClose className="w-4 h-4" />
            </button>
          </div>
          {roleLabel && (
            <span className={`inline-flex mt-2 ml-8 items-center px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase rounded-full ${roleLabel.cls}`}>
              {roleLabel.text}
            </span>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-3 flex flex-col gap-0.5 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink key={item.href} {...item} />
          ))}
        </nav>

        {/* Sign out */}
        <div className="px-3 py-3 border-t border-border">
          <button
            onClick={logout}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            <LogOut className="w-4 h-4 shrink-0" />
            Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <main className="flex-1 overflow-y-auto pb-16 md:pb-0">{children}</main>
      </div>

      {/* Mobile bottom nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-sidebar border-t border-sidebar-border flex items-stretch z-50">
        {navItems.map((item) => (
          <MobileNavLink key={item.href} {...item} />
        ))}
      </nav>
    </div>
  );
}

export default function AppShellLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthProvider>
      <AppShellInner>{children}</AppShellInner>
    </AuthProvider>
  );
}
