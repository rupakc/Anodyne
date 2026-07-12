import { auth, signOut } from "@/auth";
import { AppNav } from "@/components/app-nav";

/**
 * Shell for the whole authenticated app surface (`/app/*`, protected by
 * proxy.ts). Renders the persistent navigation above every page and wires
 * the sign-out server action once, centrally.
 */
export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();

  async function signOutAction() {
    "use server";
    await signOut({ redirectTo: "/login" });
  }

  const userLabel = session?.user?.name ?? session?.user?.email ?? "Signed in";

  return (
    <div className="flex min-h-full flex-col">
      <AppNav userLabel={userLabel} signOutAction={signOutAction} />
      <div className="flex-1">{children}</div>
    </div>
  );
}
