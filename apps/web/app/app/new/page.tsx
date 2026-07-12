import { auth } from "@/auth";
import { GenerateChooser } from "./generate-chooser";

/**
 * Generation entry point. Protected by proxy.ts (matcher: /app/:path*) and
 * wrapped by app/app/layout.tsx. Reads `session.accessToken` once and hands
 * it to the client-side chooser, which routes into the per-modality flows.
 */
export default async function NewDatasetPage() {
  const session = await auth();

  return (
    <main className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col px-6 py-10">
      <GenerateChooser accessToken={session?.accessToken} />
    </main>
  );
}
