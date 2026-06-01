"use client";

// Minimal role-aware guard + helper.
//
// Reads the role from the existing auth provider (`useAuth()` → session user)
// and gates children behind a capability. Use it to show/hide nav entries,
// buttons, or whole sections without scattering role checks across the tree.

import type { ReactNode } from "react";
import { useAuth } from "@/components/auth-provider";
import {
  capabilitiesFor,
  getRole,
  type AppRole,
  type RoleCapabilities,
} from "@/lib/roles";

/** Hook: resolve the current role + capability map from the session. */
export function useRole(): { role: AppRole; can: RoleCapabilities } {
  const { user } = useAuth();
  return { role: getRole(user), can: capabilitiesFor(user) };
}

interface RoleGuardProps {
  /** Capability the user must have. */
  require: keyof RoleCapabilities;
  children: ReactNode;
  /** Rendered when the user lacks the capability (default: nothing). */
  fallback?: ReactNode;
}

/** Render children only when the session user has the required capability. */
export function RoleGuard({ require, children, fallback = null }: RoleGuardProps) {
  const { can } = useRole();
  return <>{can[require] ? children : fallback}</>;
}
