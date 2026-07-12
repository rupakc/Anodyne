import { auth } from "@/auth";
import { JobProgressView } from "./job-progress-view";

/**
 * Live progress screen the create-from-description wizard (Task 12)
 * `router.push`es to after `POST /datasets/{id}/generate`. Protected by
 * proxy.ts (matcher: /app/:path*).
 *
 * Server component: reads `session.accessToken` once and hands it to the
 * client-side `JobProgressView`, which opens the WS stream / builds the
 * typed gateway client from it.
 */
export default async function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const session = await auth();

  return (
    <main className="mx-auto flex min-h-full w-full max-w-3xl flex-1 flex-col px-6 py-12">
      <JobProgressView jobId={id} accessToken={session?.accessToken} />
    </main>
  );
}
