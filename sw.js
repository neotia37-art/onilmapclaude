/* 오닐 터미널 — 오프라인 캐시 + 알림 */
const V = 'oneil-v1';
const SHELL = ['./', './index.html', './manifest.webmanifest'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(V).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks =>
    Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))
  ).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  // 데이터는 네트워크 우선(최신), 실패 시 캐시 — 인터넷 끊겨도 마지막 데이터 표시
  if (u.pathname.includes('/data/')) {
    e.respondWith(
      fetch(e.request).then(r => {
        const cp = r.clone();
        caches.open(V).then(c => c.put(e.request, cp));
        return r;
      }).catch(() => caches.match(e.request))
    );
    return;
  }
  // 앱 껍데기는 캐시 우선
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request).then(res => {
    const cp = res.clone();
    caches.open(V).then(c => c.put(e.request, cp));
    return res;
  })));
});
self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(clients.matchAll({type:'window', includeUncontrolled:true}).then(ws => {
    for (const w of ws) if ('focus' in w) return w.focus();
    if (clients.openWindow) return clients.openWindow('./index.html');
  }));
});
