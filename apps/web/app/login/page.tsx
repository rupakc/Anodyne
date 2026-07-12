import { signIn } from "@/auth";
import { Button } from "@/components/ui/button";

/**
 * Sign-in entry point. Auth.js's `pages.signIn` config (see auth.ts) points
 * here instead of the default unstyled /api/auth/signin page.
 *
 * The "Sign in" button posts to a Server Action that calls `signIn`
 * directly (no client-side JS needed to kick off the Keycloak redirect).
 */
export default function LoginPage() {
  async function signInWithKeycloak() {
    "use server";
    await signIn("keycloak", { redirectTo: "/app" });
  }

  return (
    <main className="flex min-h-full flex-1 items-center justify-center bg-background px-6 py-16 text-foreground">
      <div className="w-full max-w-sm rounded-2xl border border-border bg-card p-8 text-center shadow-sm">
        <p className="mb-2 font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Anodyne
        </p>
        <h1 className="font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
          Sign in to continue
        </h1>
        <p className="mt-3 text-sm text-muted-foreground text-pretty">
          Authenticate with your organization&apos;s Keycloak account to
          access the generation workspace.
        </p>
        <form action={signInWithKeycloak} className="mt-8">
          <Button type="submit" size="lg" className="w-full">
            Sign in with Keycloak
          </Button>
        </form>
      </div>
    </main>
  );
}
