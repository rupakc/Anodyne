import { auth, signOut } from "@/auth";
import { Button } from "@/components/ui/button";

/**
 * Placeholder for the authenticated app workspace. Protected by proxy.ts
 * (matcher: /app/:path*) — unauthenticated requests never reach this
 * component; they're redirected to /login first.
 */
export default async function AppHome() {
  const session = await auth();

  async function logout() {
    "use server";
    await signOut({ redirectTo: "/login" });
  }

  return (
    <main className="flex min-h-full flex-1 flex-col items-center justify-center gap-4 bg-background px-6 py-16 text-foreground">
      <p className="font-[family-name:var(--font-display)] text-xl font-semibold tracking-tight">
        Signed in as {session?.user?.name ?? session?.user?.email ?? "you"}
      </p>
      <p className="max-w-md text-center text-sm text-muted-foreground text-pretty">
        This is a placeholder for the generation workspace. The gateway API
        client reads <code className="font-[family-name:var(--font-data)]">session.accessToken</code>{" "}
        from here to call <code className="font-[family-name:var(--font-data)]">/llm/invoke</code> and friends.
      </p>
      <form action={logout}>
        <Button type="submit" variant="outline">
          Sign out
        </Button>
      </form>
    </main>
  );
}
