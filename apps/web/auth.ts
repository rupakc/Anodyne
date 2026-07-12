import NextAuth, { type NextAuthConfig } from "next-auth";
import Keycloak from "next-auth/providers/keycloak";

/**
 * Auth.js (NextAuth v5) config wired to the local Keycloak realm.
 *
 * Env vars (see docs/dev-runbook.md for the full list and how to set them
 * in `apps/web/.env.local`):
 * - KEYCLOAK_ISSUER       (default: http://localhost:8080/realms/anodyne)
 * - KEYCLOAK_CLIENT_ID    (default: anodyne)
 * - KEYCLOAK_CLIENT_SECRET (required — the realm's dev client secret; see
 *   infra/docker/keycloak/anodyne-realm.json for the dev-only value)
 * - AUTH_SECRET           (required — `openssl rand -base64 32`)
 *
 * `authConfig` is exported separately (rather than inlined into the
 * `NextAuth(...)` call) so `__tests__/auth-callbacks.test.ts` can import and
 * exercise the `jwt`/`session` callback functions directly, without a live
 * Keycloak instance.
 */
export const authConfig = {
  providers: [
    Keycloak({
      issuer: process.env.KEYCLOAK_ISSUER ?? "http://localhost:8080/realms/anodyne",
      clientId: process.env.KEYCLOAK_CLIENT_ID ?? "anodyne",
      clientSecret: process.env.KEYCLOAK_CLIENT_SECRET,
    }),
  ],
  pages: {
    // Custom autumn-pastel styled sign-in entry (app/login/page.tsx),
    // in place of Auth.js's default unstyled /api/auth/signin page.
    signIn: "/login",
  },
  callbacks: {
    // Stash the Keycloak access token on the JWT right after sign-in, when
    // `account` is populated (it's only present on the initial sign-in
    // request, not on subsequent session reads).
    jwt({ token, account }) {
      if (account?.access_token) {
        token.accessToken = account.access_token;
      }
      return token;
    },
    // Surface the access token on the session so client/server code can
    // attach it as `Authorization: Bearer <token>` when calling the gateway.
    session({ session, token }) {
      session.accessToken = token.accessToken;
      return session;
    },
    // Gates which routes require a session; see proxy.ts for the matcher
    // that decides which requests this even runs for.
    authorized({ auth }) {
      return !!auth;
    },
  },
} satisfies NextAuthConfig;

export const { handlers, auth, signIn, signOut } = NextAuth(authConfig);
