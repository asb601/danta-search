"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft, LogOut, UserCircle, Database, ScrollText,
  Tag, Users, Shield, Sun, Moon,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "@/components/auth-provider";
import { useTheme } from "@/components/theme-provider";
import { cn } from "@/lib/utils";
import { capabilitiesFor } from "@/lib/roles";

// Embedded page content — these are self-contained client components
import ContainersPage from "../admin/containers/page";
import AdminLogsPage from "../admin/logs/page";

// Profile, Users, Domains, Parquet — pull from profile page exports
import ProfileTabContent from "./_components/ProfileTabContent";
import UsersTabContent from "./_components/UsersTabContent";
import DomainsTabContent from "./_components/DomainsTabContent";

/* ── types ───────────────────────────────────────────────────────────────── */

type Tab = "profile" | "users" | "domains" | "containers" | "logs";

/* ── page ────────────────────────────────────────────────────────────────── */

export default function SettingsPage() {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("profile");

  if (!user) return null;

  // /users/pending + /users/{id}/grant require platform admin; a non-platform
  // is_admin would hit 403, so gate the Users tab on platform-admin capability.
  const isPlatformAdmin = capabilitiesFor(user).canManageOrganizations;

  const tabs: {
    id: Tab;
    label: string;
    icon: typeof UserCircle;
    adminOnly?: boolean;
    platformAdminOnly?: boolean;
  }[] = [
    { id: "profile",    label: "Profile",    icon: UserCircle },
    { id: "users",      label: "Users",      icon: Users,      platformAdminOnly: true },
    { id: "domains",    label: "Domains",    icon: Tag,        adminOnly: true },
    { id: "containers", label: "Containers", icon: Database,   adminOnly: true },
    { id: "logs",       label: "Logs",       icon: ScrollText },
  ];

  const visibleTabs = tabs.filter(
    (t) =>
      (!t.adminOnly || user.is_admin) &&
      (!t.platformAdminOnly || isPlatformAdmin)
  );

  return (
    <div className="flex flex-col h-full bg-white">

      {/* ── Header ───────────────────────────────────────────────── */}
      <div className="shrink-0 px-5 sm:px-6 pt-5 pb-0 border-b border-[#e5e5e5]">
        {/* Top row: back + title + sign out */}
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <motion.button
              whileHover={{ x: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => router.push("/chat")}
              className="flex items-center gap-1.5 text-[12px] text-[#a3a3a3] hover:text-[#0a0a0a] transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Chat</span>
            </motion.button>
            <div className="w-px h-3.5 bg-[#e5e5e5]" />
            <h1
              className="text-[17px] font-bold text-[#0a0a0a]"
              style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}
            >
              Settings
            </h1>
            {user.is_admin && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-[#0a0a0a] text-white">
                <Shield className="w-2.5 h-2.5" />
                Admin
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Theme toggle */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.93 }}
              onClick={toggleTheme}
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              className="relative w-14 h-7 rounded-full border border-[#e0dfd9] bg-[#f1f0ec] transition-colors overflow-hidden"
              style={theme === "dark" ? { background: "#272725", borderColor: "#2e2d2a" } : {}}
            >
              {/* Track icons */}
              <Sun className="absolute left-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-amber-500" />
              <Moon className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-indigo-400" />
              {/* Thumb */}
              <motion.div
                animate={{ x: theme === "dark" ? 28 : 2 }}
                transition={{ type: "spring", stiffness: 500, damping: 36 }}
                className="absolute top-1 w-5 h-5 rounded-full bg-white shadow-[0_1px_4px_rgba(0,0,0,0.18)]"
                style={theme === "dark" ? { background: "#f0efe9" } : {}}
              />
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.96 }}
              onClick={logout}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[12px] font-medium text-[#dc2626] bg-[#dc2626]/6 hover:bg-[#dc2626]/12 border border-[#dc2626]/15 transition-colors"
            >
              <LogOut className="w-3.5 h-3.5" />
              Sign out
            </motion.button>
          </div>
        </div>

        {/* Tab bar */}
        <div className="flex gap-0.5 overflow-x-auto" style={{ scrollbarWidth: "none" }}>
          {visibleTabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "relative flex items-center gap-1.5 px-3 py-2 text-[13px] font-medium transition-colors rounded-t-lg whitespace-nowrap",
                tab === t.id ? "text-[#0a0a0a]" : "text-[#a3a3a3] hover:text-[#737373]"
              )}
            >
              <t.icon className="w-3.5 h-3.5" />
              {t.label}
              {tab === t.id && (
                <motion.div
                  layoutId="settings-tab-indicator"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#0a0a0a] rounded-full"
                  transition={{ type: "spring", stiffness: 420, damping: 36 }}
                />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* ── Content ──────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className={cn(tab === "containers" || tab === "logs" ? "" : "p-5 sm:p-6")}
          >
            {tab === "profile"    && <ProfileTabContent />}
            {tab === "users"      && isPlatformAdmin && <UsersTabContent currentUserId={user.id} />}
            {tab === "domains"    && user.is_admin && <DomainsTabContent />}
            {tab === "containers" && <ContainersPage />}
            {tab === "logs"       && <AdminLogsPage />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
