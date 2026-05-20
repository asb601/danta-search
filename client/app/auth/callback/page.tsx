"use client";

import { Suspense, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { setToken } from "@/lib/auth";

function CallbackHandler() {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const token = searchParams.get("token");
    if (!token) {
      router.replace("/login?error=no_token");
      return;
    }
    setToken(token);
    const secure = window.location.protocol === "https:" ? "; Secure" : "";
    document.cookie = `token=${encodeURIComponent(token)}; path=/; max-age=${60 * 60 * 24 * 7}; SameSite=Lax${secure}`;
    // Navigate immediately — AuthProvider on /chat validates the token
    // and handles onboarding redirect if needed.
    window.location.replace("/chat");
  }, [searchParams, router]);

  return null;
}

export default function AuthCallbackPage() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <Suspense fallback={<p className="text-muted-foreground text-sm">Signing you in…</p>}>
        <CallbackHandler />
      </Suspense>
    </div>
  );
}
