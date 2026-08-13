# Frontend — Sumina

Single-page React app ("Sumina") for uploading a scan, reviewing/correcting the
SuminaNet transcription, and getting an English or German translation.

## Stack

| Layer | Choice |
|---|---|
| Framework | [React](https://react.dev/) 19 |
| Build tool / dev server | [Vite](https://vitejs.dev/) 8 (`@vitejs/plugin-react`) |
| Routing | `react-router` 8 |
| i18n | `i18next` — locale strings in [`locales/en/`](locales/en/) and [`locales/de/`](locales/de/) |
| Styling | Plain CSS ([`style.css`](style.css)), `flag-icons` for language flags |
| Runtime | Node.js 22+ (for building; no Node runtime needed to serve the built output) |

No client-side state management library (Redux/Zustand) — component state only.
No TypeScript — plain `.jsx`.

## Structure

```
pages/        Home, Workspace (upload+transcribe+translate), TranscriptionPanel,
               About, Policy — one component per route
components/    UploadArea, ImageWithBoxes (renders char bounding boxes over the
               scan), LoadingIndicator
locales/       en/, de/ translation.json — UI strings + policy/about copy
utils/text.js  text helpers (furigana stripping, etc.)
utils/api.js   apiFetch() — wraps fetch() with the backend base URL + API key
```

## Talking to the backend

Both API calls in [Workspace.jsx](pages/Workspace.jsx) go through
`apiFetch()` in [`utils/api.js`](utils/api.js), which reads two build-time env
vars (see [`.env.example`](.env.example)):

- `VITE_API_BASE` — prefixed onto the request path. Empty by default, meaning
  requests go to relative paths (`/api/transcribe`) on the frontend's own origin.
- `VITE_API_KEY` — sent as the `X-API-Key` header when set. Must match the
  backend's `API_KEY` (see `../../.env.example`). Omitted from requests entirely
  when unset.

With both unset (the default), behavior is unchanged from before: relative
`/api/*` paths, no extra header.

- **Dev**: [`vite.config.js`](vite.config.js) proxies `/api/*` to
  `http://localhost:8000` (the FastAPI dev server); no env vars needed.
- **Production, same origin**: leave `VITE_API_BASE` unset and route `/api/*` to
  the backend at the reverse proxy / static host level.
- **Production, different origin**: set `VITE_API_BASE` to the backend's URL at
  build time. Note this is baked into the static bundle — changing it requires a
  rebuild, not just a redeploy of the same `dist/`.

`VITE_API_KEY`, if set, ends up readable in the built JS bundle (it's a public
SPA, not a secret server) — see the caveat in
[`app/backend/README.md`](../backend/README.md#configuration).

## Building

```bash
cd app/frontend
npm install
npm run build      # outputs static files to dist/
npm run preview    # serve the production build locally for a smoke test
```

`dist/` is a static bundle (~700 KB uncompressed as of the last build) — no Node
server is required to serve it. Deploy it behind any static file host / CDN / nginx,
with `/api/*` routed to the FastAPI backend as described above.

## Dev server

```bash
npm run dev   # http://localhost:5173, proxies /api to :8000
```

The backend's default CORS allowlist (`CORS_ORIGINS` in `.env`) only permits
`http://localhost:5173` / `127.0.0.1:5173` out of the box — set `CORS_ORIGINS` on the
backend to the real frontend origin before deploying.
