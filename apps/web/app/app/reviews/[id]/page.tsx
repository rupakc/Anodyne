import { auth } from "@/auth";
import { ReviewDetail } from "./review-detail";

/**
 * Single review detail + decision. Protected by proxy.ts; wrapped by
 * app/app/layout.tsx.
 */
export default async function ReviewDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const session = await auth();
  return (
    <main className="mx-auto flex min-h-full w-full max-w-3xl flex-1 flex-col px-6 py-10">
      <ReviewDetail reviewId={id} accessToken={session?.accessToken} />
    </main>
  );
}
