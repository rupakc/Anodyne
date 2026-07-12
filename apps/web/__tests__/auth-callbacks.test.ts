import { describe, expect, it } from "vitest";
import type { Account } from "next-auth";
import type { JWT } from "next-auth/jwt";

import { authConfig } from "@/auth";

const { jwt, session } = authConfig.callbacks;

describe("auth.ts callbacks", () => {
  describe("jwt callback", () => {
    it("stores account.access_token on the token on initial sign-in", async () => {
      const token: JWT = {};
      const account = {
        access_token: "keycloak-access-token",
        provider: "keycloak",
        type: "oidc",
        providerAccountId: "user-123",
      } as Account;

      // @ts-expect-error -- exercising the callback with the minimal shape
      // it actually reads from; the full NextAuth callback signature takes
      // several more (unused-here) params.
      const result = await jwt({ token, account });

      expect(result.accessToken).toBe("keycloak-access-token");
    });

    it("leaves an existing token.accessToken untouched when account is absent", async () => {
      const token: JWT = { accessToken: "previously-stored-token" };

      // @ts-expect-error -- see above
      const result = await jwt({ token, account: null });

      expect(result.accessToken).toBe("previously-stored-token");
    });
  });

  describe("session callback", () => {
    it("surfaces token.accessToken on session.accessToken", async () => {
      const token: JWT = { accessToken: "keycloak-access-token" };
      const baseSession = {
        user: { name: "Demo User" },
        expires: "2099-01-01T00:00:00.000Z",
      };

      // @ts-expect-error -- see above
      const result = await session({ session: baseSession, token });

      expect(result.accessToken).toBe("keycloak-access-token");
    });
  });
});
