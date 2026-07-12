import type { DefaultSession } from "next-auth";

// Module augmentation: the `jwt`/`session` callbacks in `auth.ts` copy the
// Keycloak access token onto the token and then the session, so downstream
// code (e.g. the API client calling the gateway) can read
// `session.accessToken` and send it as `Authorization: Bearer <token>`.
declare module "next-auth" {
  interface Session extends DefaultSession {
    accessToken?: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string;
  }
}
