import { auth } from "@/auth";
import { ReviewQueue } from "./review-queue";

/**
 * HITL review queue. Protected by proxy.ts and wrapped by app/app/layout.tsx.
 */
export default async function ReviewsPage() {
  const session = await auth();
  return (
    <main className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col px-6 py-10">
      <ReviewQueue accessToken={session?.accessToken} />
    </main>
  );
}
