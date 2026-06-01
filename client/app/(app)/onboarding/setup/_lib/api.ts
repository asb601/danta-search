// Shared helpers for the onboarding wizard's API calls.

/** Extract a human-readable error message from a failed Response. */
export async function safeError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (Array.isArray(data?.detail) && data.detail[0]?.msg)
      return data.detail[0].msg as string;
    if (typeof data?.message === "string") return data.message;
  } catch {
    /* not JSON */
  }
  return `Request failed (${res.status}).`;
}
