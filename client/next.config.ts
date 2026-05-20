import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // Proxy all /api/* calls through Vercel so browsers (e.g. Brave) never
    // see a cross-origin request — they only talk to this same domain.
    const backendUrl = (
      process.env.BACKEND_URL
      ?? process.env.NEXT_PUBLIC_API_URL
      ?? "http://localhost:8000"
    ).replace(/\/$/, "");
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
