import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits `.next/standalone` (a self-contained `server.js` + pruned
  // node_modules) so the production Docker image (apps/web/Dockerfile) can
  // ship a minimal runtime stage without `node_modules`/pnpm at all. See
  // docs/deployment.md. Pure build-output config — no behavior change for
  // `next dev` or `next start` in local/dev use.
  output: "standalone",
};

export default nextConfig;
