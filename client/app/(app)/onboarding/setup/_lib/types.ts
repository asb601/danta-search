// Onboarding wizard contract — mirrors the backend (LANE for /api/onboarding/*).
// Code defensively: the backend is built in parallel, so treat all fields as
// optional and tolerate partial state.

export type StepKey =
  | "owner_signin"
  | "ai_settings"
  | "storage"
  | "domains"
  | "users"
  | "access_control"
  | "complete";

/** Ordered step definitions used to render the wizard + checklist. */
export const STEP_ORDER: StepKey[] = [
  "owner_signin",
  "ai_settings",
  "storage",
  "domains",
  "users",
  "access_control",
];

export interface StepMeta {
  key: StepKey;
  title: string;
  shortTitle: string;
  description: string;
  optional?: boolean;
}

export const STEPS: StepMeta[] = [
  {
    key: "owner_signin",
    title: "Owner sign-in",
    shortTitle: "Sign in",
    description: "Sign in with the owner Google account to begin setup.",
  },
  {
    key: "ai_settings",
    title: "AI settings",
    shortTitle: "AI settings",
    description: "Configure the API keys that power chat and embeddings.",
  },
  {
    key: "storage",
    title: "Connect storage",
    shortTitle: "Storage",
    description: "Connect the Azure Blob Storage account for this organization.",
  },
  {
    key: "domains",
    title: "Create domains",
    shortTitle: "Domains",
    description: "Define the data domains users will be scoped to.",
  },
  {
    key: "users",
    title: "User management",
    shortTitle: "Users",
    description: "Invite teammates individually or in bulk via Excel.",
  },
  {
    key: "access_control",
    title: "Access control",
    shortTitle: "Access",
    description: "Optionally grant Platform Admin access to your organization.",
    optional: true,
  },
];

/** Server-side onboarding state returned by GET /api/onboarding/state. */
export interface OnboardingState {
  /** The step the wizard should currently be on. */
  current_step?: StepKey | null;
  /** Steps that are finished (used to gate forward navigation + checklist). */
  completed_steps?: StepKey[] | null;
  /** Whether onboarding has been fully completed. */
  completed?: boolean | null;
  /** Convenience pre-filled values (so re-entry shows prior input). */
  ai_settings?: {
    chat_endpoint?: string | null;
    embeddings_endpoint?: string | null;
    chat_deployment?: string | null;
    api_version?: string | null;
    postgres_url?: string | null;
    configured?: boolean | null;
  } | null;
  storage?: {
    container_name?: string | null;
    configured?: boolean | null;
  } | null;
  domains?: { id?: string; name: string; label?: string | null }[] | null;
  users?: {
    email: string;
    role: string;
    domains?: string[] | null;
  }[] | null;
  platform_admin_granted?: boolean | null;
}

export interface OnboardingDomain {
  id?: string;
  name: string;
  label?: string | null;
}

export interface OnboardingUser {
  email: string;
  role: "admin" | "manager" | "user";
  domains?: string[] | null;
}
