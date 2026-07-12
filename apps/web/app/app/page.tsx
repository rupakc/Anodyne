import { auth } from "@/auth";
import { Dashboard } from "./dashboard";

/**
 * Tenant home / dashboard. Protected by proxy.ts (matcher: /app/:path*) and
 * wrapped by app/app/layout.tsx (which provides the nav + sign-out). Reads
 * `session.accessToken` once and hands it to the client `Dashboard`.
 */
export default async function AppHome() {
  const session = await auth();
  return <Dashboard accessToken={session?.accessToken} />;
}
