import { auth } from "@/auth";
import { Wizard } from "./wizard";

/**
 * Create-from-description entry point. Protected by proxy.ts (matcher:
 * /app/:path*), same as the rest of the authenticated app surface.
 *
 * Server component: it reads `session.accessToken` once (Task 11's OIDC
 * flow) and hands it to the client-side `Wizard`, which builds the typed
 * gateway client (`lib/api.ts`) from it.
 */
export default async function NewDatasetPage() {
  const session = await auth();

  return (
    <main className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col px-6 py-12">
      <div className="mb-10">
        <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Anodyne
        </p>
        <h1 className="mt-2 font-[family-name:var(--font-display)] text-3xl font-semibold tracking-tight text-balance">
          Create a dataset from a description
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground text-pretty">
          Describe what you need in plain English, review the schema the
          gateway proposes, then generate real synthetic rows.
        </p>
      </div>
      <Wizard accessToken={session?.accessToken} />
    </main>
  );
}
