import { auth } from "@/auth";
import { PerturbationView } from "./perturbation-view";

/**
 * Live progress for a perturbation run, reached from a version's "Perturb"
 * action. Protected by proxy.ts (matcher: /app/:path*).
 */
export default async function PerturbationPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const session = await auth();

  return (
    <main className="mx-auto flex min-h-full w-full max-w-3xl flex-1 flex-col px-6 py-10">
      <PerturbationView jobId={id} accessToken={session?.accessToken} />
    </main>
  );
}
