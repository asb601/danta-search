// Role-aware capability helpers.
//
// Roles are read from the existing auth/session source (`User` from lib/auth,
// surfaced via the `useAuth()` provider). The backend exposes `is_admin`
// (platform-level) and a string `role`. We normalize those into a single
// `AppRole` plus a capability map the UI can branch on.
//
// Role hierarchy (highest → lowest capability):
//   platform_admin   — manage organizations (cross-tenant)
//   org_owner        — full control of their organization (Google-SSO owner)
//   org_admin        — full control of their organization
//   manager          — their assigned domains only (can manage domain users)
//   user             — their assigned domains only (read/chat)

import type { User } from "@/lib/auth";

export type AppRole =
  | "platform_admin"
  | "org_owner"
  | "org_admin"
  | "manager"
  | "user";

export interface RoleCapabilities {
  /** Cross-tenant: manage organizations. */
  canManageOrganizations: boolean;
  /** Full control of the current organization (settings, storage, AI, users). */
  canManageOrg: boolean;
  /** Run / complete the organization onboarding wizard. */
  canRunOnboarding: boolean;
  /** Manage users within the org / assigned domains. */
  canManageUsers: boolean;
  /** Manage data containers. */
  canManageContainers: boolean;
  /** Scoped to assigned domains only (no org-wide visibility). */
  domainScopedOnly: boolean;
}

/** Resolve the canonical app role from the session user. */
export function getRole(user: Pick<User, "is_admin" | "role"> | null): AppRole {
  if (!user) return "user";
  const raw = (user.role ?? "").toLowerCase();

  // Explicit platform-level flags / roles win.
  if (raw === "platform_admin") return "platform_admin";
  if (user.is_admin && (raw === "" || raw === "admin" || raw === "developer")) {
    return "platform_admin";
  }
  if (raw === "org_owner" || raw === "owner") return "org_owner";
  if (raw === "org_admin" || raw === "admin") return "org_admin";
  if (raw === "manager") return "manager";
  return "user";
}

/** Capability map for a role. */
export function getCapabilities(role: AppRole): RoleCapabilities {
  switch (role) {
    case "platform_admin":
      return {
        canManageOrganizations: true,
        canManageOrg: true,
        canRunOnboarding: true,
        canManageUsers: true,
        canManageContainers: true,
        domainScopedOnly: false,
      };
    case "org_owner":
    case "org_admin":
      return {
        canManageOrganizations: false,
        canManageOrg: true,
        canRunOnboarding: true,
        canManageUsers: true,
        canManageContainers: true,
        domainScopedOnly: false,
      };
    case "manager":
      return {
        canManageOrganizations: false,
        canManageOrg: false,
        canRunOnboarding: false,
        canManageUsers: true,
        canManageContainers: false,
        domainScopedOnly: true,
      };
    case "user":
    default:
      return {
        canManageOrganizations: false,
        canManageOrg: false,
        canRunOnboarding: false,
        canManageUsers: false,
        canManageContainers: false,
        domainScopedOnly: true,
      };
  }
}

/** Convenience: capabilities directly from a session user. */
export function capabilitiesFor(
  user: Pick<User, "is_admin" | "role"> | null,
): RoleCapabilities {
  return getCapabilities(getRole(user));
}

/** Human-readable label + badge classes for a role (matches app shell badges). */
export function roleBadge(
  role: AppRole,
): { text: string; cls: string } | null {
  switch (role) {
    case "platform_admin":
      return { text: "Platform Admin", cls: "bg-[#0a0a0a]/8 text-[#0a0a0a]" };
    case "org_owner":
      return { text: "Owner", cls: "bg-violet-500/10 text-violet-600" };
    case "org_admin":
      return { text: "Admin", cls: "bg-[#0a0a0a]/8 text-[#0a0a0a]" };
    case "manager":
      return { text: "Manager", cls: "bg-cyan-500/10 text-cyan-700" };
    case "user":
    default:
      return null;
  }
}
