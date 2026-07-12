# Multi-Tenancy & Security

Anodyne is **multi-tenant**: many companies (tenants) use the same platform, but each one's data is
completely walled off from the others. This page explains how that isolation works, in plain terms.

## The core idea

Every piece of data is stamped with the company it belongs to. Anodyne enforces "you can only ever
see your own company's data" at **several layers**, so a mistake in one layer can't cause a leak.

## The layers of protection

1. **Login & identity.** You sign in through your organization; your identity carries which company
   (tenant) you belong to.
2. **Database-level isolation (the strongest guard).** The database itself refuses to return another
   company's rows — a feature called *row-level security*. Even if application code had a bug, the
   database won't hand over data that isn't yours. Anodyne runs as a restricted database user
   specifically so this rule always applies.
3. **File storage isolation.** Generated files are stored under a per-company folder, so one
   company's files never mix with another's.
4. **Permission checks.** Every action checks your role (owner/admin/member/viewer) before it runs.
5. **Secret protection.** AI provider keys are encrypted at rest and never displayed or logged.

## How we know it works

The database isolation isn't just claimed — it's **automatically tested** against a real database:
the tests create two companies, add data as one, and verify the other genuinely cannot see it. These
tests run in the automated pipeline on every change.

## Status

✅ Built and verified. This protection covers all data the [Generation Engine](Generation-Engine)
creates (datasets, jobs, versions) and every earlier feature.
