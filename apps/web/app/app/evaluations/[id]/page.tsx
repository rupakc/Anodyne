import { auth } from "@/auth";
import { EvaluationView } from "./evaluation-view";

/**
 * Evaluation run + report screen. Reached from a dataset version's "Evaluate"
 * action, which launches the run and routes here. Protected by proxy.ts
 * (matcher: /app/:path*).
 */
export default async function EvaluationPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const session = await auth();

  return (
    <main className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col px-6 py-10">
      <EvaluationView evaluationId={id} accessToken={session?.accessToken} />
    </main>
  );
}
