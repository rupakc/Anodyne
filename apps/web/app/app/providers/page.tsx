import { auth } from "@/auth";
import { ProvidersManager } from "./providers-manager";

/**
 * Provider registry management. Protected by proxy.ts (matcher: /app/:path*)
 * and wrapped by app/app/layout.tsx. Reads `session.accessToken` once and
 * hands it to the client `ProvidersManager`.
 */
export default async function ProvidersPage() {
  const session = await auth();

  return (
    <main className="mx-auto flex min-h-full w-full max-w-5xl flex-1 flex-col px-6 py-10">
      <ProvidersManager accessToken={session?.accessToken} />
    </main>
  );
}
