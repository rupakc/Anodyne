# syntax=docker/dockerfile:1
#
# web — Next.js (App Router) frontend, built for the standalone output
# (`output: "standalone"` in apps/web/next.config.ts) so the runtime stage
# needs neither `node_modules` nor pnpm.
#
# Build from the REPO ROOT so the .dockerignore rules stay consistent with
# the other two images:
#   docker build -f apps/web/Dockerfile -t anodyne/web .
FROM node:22-slim AS deps

# Corepack ships with Node >=16.9 but needs enabling; once enabled, running
# `pnpm` at all auto-installs+uses the exact version pinned by
# apps/web/package.json's `packageManager` field (no separate `corepack use`
# call needed).
RUN corepack enable
WORKDIR /app/apps/web
COPY apps/web/package.json apps/web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM node:22-slim AS builder
RUN corepack enable
WORKDIR /app/apps/web
COPY --from=deps /app/apps/web/node_modules ./node_modules
COPY apps/web ./
# Build-time-only public config (baked into the client bundle); real
# deployments set this via the platform's build config (Cloud Run
# build-args / Kustomize overlay), see docs/deployment.md.
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}
RUN pnpm build

FROM node:22-slim AS runtime
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0
WORKDIR /app

# node:22-slim's Debian base image already ships a non-root `node` user
# (uid/gid 1000); standalone output's server.js is self-contained, so the
# runtime stage carries no pnpm/node_modules at all.
#
# apps/web has its own pnpm-lock.yaml (no root pnpm-workspace.yaml/turbo.json
# in this repo), so Next's output-file-tracing root is apps/web itself and
# `.next/standalone/server.js` lands un-nested — copy it straight to /app,
# then lay `.next/static` and `public/` alongside it, exactly as the
# standalone server expects (see Next.js "output: standalone" docs).
COPY --from=builder --chown=node:node /app/apps/web/.next/standalone ./
COPY --from=builder --chown=node:node /app/apps/web/.next/static ./.next/static
COPY --from=builder --chown=node:node /app/apps/web/public ./public

USER node
EXPOSE 3000
CMD ["node", "server.js"]
