import { handlers } from "@/auth";

// Auth.js's catch-all route handler: serves the sign-in/callback/session
// endpoints under /api/auth/* (e.g. /api/auth/signin, /api/auth/callback/keycloak).
export const { GET, POST } = handlers;
