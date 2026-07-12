// Next.js 16 renamed `middleware.ts` to `proxy.ts` (same functionality,
// runs on the nodejs runtime — see node_modules/next/dist/docs/01-app/
// 02-guides/upgrading/version-16.md). We use the `auth` wrapper directly:
// it runs the `authorized` callback from auth.ts, and on failure redirects
// unauthenticated requests to the sign-in page configured there (/login).
export { auth as proxy } from "@/auth";

export const config = {
  // Only the authenticated app surface requires a session; the marketing
  // page, the login page, and the /api/auth/* Auth.js routes stay public.
  matcher: ["/app/:path*"],
};
