/* Boatrace Lab service worker
   静的資産: Cache First（バージョン付き）
   API     : Network First。失敗時のみキャッシュを返し、X-From-Cache ヘッダで「古い可能性」を伝える。
   予想・オッズは常に最新をネットワークから取り、古いキャッシュを最新として見せない（docs/06 §3）。 */
const VERSION = "bl-v2";
const STATIC = ["/", "/static/manifest.json", "/static/icon.svg"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(STATIC)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  if (url.pathname.startsWith("/api/")) {
    e.respondWith(
      fetch(e.request).then((res) => {
        const copy = res.clone();
        caches.open(VERSION).then((c) => c.put(e.request, copy));
        return res;
      }).catch(async () => {
        const cached = await caches.match(e.request);
        if (!cached) return new Response(JSON.stringify({ offline: true }), { status: 503, headers: { "Content-Type": "application/json" } });
        const h = new Headers(cached.headers); h.set("X-From-Cache", "1");
        return new Response(await cached.blob(), { status: cached.status, headers: h });
      })
    );
    return;
  }
  e.respondWith(caches.match(e.request).then((c) => c || fetch(e.request)));
});
