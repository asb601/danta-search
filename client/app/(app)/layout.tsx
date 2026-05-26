"use client";

import { useRouter, usePathname } from "next/navigation";
import { useState, useEffect, useCallback } from "react";
import { MessageSquare, FolderOpen, LogOut, PanelLeftClose, PanelLeft, Database, UserCircle, ScrollText } from "lucide-react";
import { NavLink, MobileNavLink } from "@/components/nav-link";
import { AuthProvider, useAuth } from "@/components/auth-provider";
import { useIdleTimeout } from "@/hooks/use-idle-timeout";

const IDLE_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes

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
  const hideNav=noNavRoutes.includes(pathname);

  const handleIdle = useCallback(() => {
    logout();
    router.replace("/login");
  }, [logout, router]);

  useIdleTimeout({ timeoutMs: IDLE_TIMEOUT_MS, onTimeout: handleIdle });

  useEffect(() => {
    if (loading) return;
    if (!user) {
      // Clear the token cookie so the middleware doesn't bounce the user
      // back to /chat after we redirect them to /login.
      document.cookie = "token=; path=/; max-age=0";
      router.replace("/login");
      return;
    }
    // Regular users with no domain selection should be routed to onboarding
    if (!user.is_admin && !user.allowed_domains && pathname !== "/onboarding") {
      router.replace("/onboarding");
    }
  }, [loading, user, router, pathname]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    );
  }

  if (!user) {
    return null;
  }

if (hideNav) {
  return (
    <div className="h-screen bg-background">
      {children}
    </div>
  );
}
  const navItems: NavItem[] = [
    { href: "/chat", icon: MessageSquare, label: "Chat" },
    { href: "/folders", icon: FolderOpen, label: "Folders" },
    ...(user.is_admin || user.role === "developer"
      ? [{ href: "/admin/containers", icon: Database, label: "Containers" }]
      : []),
    ...(user.is_admin || user.role === "developer" || user.role === "manager" || user.role === "user"
      ? [{ href: "/admin/logs", icon: ScrollText, label: "Logs" }]
      : []),
    { href: "/profile", icon: UserCircle, label: "Profile" },
  ];

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Sidebar toggle when collapsed */}
      {!sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          className="hidden md:flex fixed top-4 left-4 z-40 items-center justify-center w-8 h-8 rounded-md bg-card border border-border shadow-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <PanelLeft className="w-4 h-4" />
        </button>
      )}

      {/* Desktop sidebar */}
      <aside
        className={`hidden md:flex flex-col shrink-0 bg-sidebar border-r border-sidebar-border h-screen sticky top-0 transition-[width] duration-200 ${sidebarOpen ? "w-[220px]" : "w-0 overflow-hidden border-r-0"}`}
      >
        <div className="px-4 py-5 border-b border-border flex items-start justify-between">
          <div className="min-w-0">
            <p className="text-sm font-semibold tracking-tight text-foreground">danta-search</p>
            <p className="text-xs text-muted-foreground mt-0.5 truncate">
              {user.email}
            </p>
            {user.is_admin && (
              <span className="inline-flex mt-1.5 items-center px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase rounded-full bg-primary/10 text-primary">
                Admin
              </span>
            )}
            {!user.is_admin && user.role === "developer" && (
              <span className="inline-flex mt-1.5 items-center px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase rounded-full bg-violet-500/10 text-violet-500">
                Developer
              </span>
            )}
            {!user.is_admin && user.role === "manager" && (
              <span className="inline-flex mt-1.5 items-center px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase rounded-full bg-cyan-500/10 text-cyan-600">
                Manager
              </span>
            )}
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="mt-0.5 p-1 rounded text-muted-foreground hover:text-foreground transition-colors"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        </div>

        <nav className="flex-1 px-3 py-4 flex flex-col gap-1">
          {navItems.map((item) => (
            <NavLink key={item.href} {...item} />
          ))}
        </nav>

        <div className="px-3 py-4 border-t border-border">
          <button
            onClick={logout}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-surface-raised transition-colors"
          >
            <LogOut className="w-4 h-4" />
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
