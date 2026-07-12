import Link from "next/link";
import { auth } from "@/auth";
import { DatasetList } from "./dataset-list";

/**
 * Dataset browser: lists every dataset the tenant has created. Protected by
 * proxy.ts (matcher: /app/:path*).
 */
export default async function DatasetsPage() {
  const session = await auth();

  return (
    <main className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col px-6 py-12">
      <div className="mb-10 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
            Anodyne
          </p>
          <h1 className="mt-2 font-[family-name:var(--font-display)] text-3xl font-semibold tracking-tight text-balance">
            Your datasets
          </h1>
        </div>
        <Link
          href="/app/new"
          className="rounded-lg bg-terracotta px-4 py-2 text-sm font-medium text-terracotta-foreground transition-colors hover:bg-terracotta/85"
        >
          + New dataset
        </Link>
      </div>
      <DatasetList accessToken={session?.accessToken} />
    </main>
  );
}
