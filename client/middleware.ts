import { NextResponse } from "next/server";

export function middleware() {
  // Auth is validated by AuthProvider using the JWT in localStorage and
  // /api/auth/me. Middleware cannot read localStorage, and cookie-only checks
  // caused login loops in browsers that delayed or blocked the companion cookie.
  return NextResponse.next();
}

export const config = {
  matcher: ["/chat/:path*", "/folders/:path*", "/admin/:path*", "/profile/:path*", "/onboarding/:path*", "/onboarding", "/login"],
};
