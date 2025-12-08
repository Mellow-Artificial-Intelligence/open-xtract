FROM oven/bun:1 AS base

# Development target
FROM base AS development
WORKDIR /app
COPY package.json bun.lock* ./
RUN bun install
COPY . .
EXPOSE 3000
ENV NODE_ENV=development
CMD ["bun", "run", "dev"]

# Build target
FROM base AS builder
WORKDIR /app
COPY package.json bun.lock* ./
RUN bun install
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN bun run build

# Production target
FROM base AS production
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["bun", "run", "server.js"]
