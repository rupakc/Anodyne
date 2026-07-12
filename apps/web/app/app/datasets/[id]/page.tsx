import { auth } from "@/auth";
import { DatasetVersions } from "./dataset-versions";

/**
 * Dataset detail: versions + download. Reached either from the dataset
 * browser (`/app/datasets`) or from a succeeded job's progress page
 * (`/app/jobs/{id}`). Protected by proxy.ts (matcher: /app/:path*).
 */
export default async function DatasetDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const session = await auth();

  return (
    <main className="mx-auto flex min-h-full w-full max-w-3xl flex-1 flex-col px-6 py-12">
      <DatasetVersions datasetId={id} accessToken={session?.accessToken} />
    </main>
  );
}
