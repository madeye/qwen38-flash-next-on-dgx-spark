#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp>=3.9"]
# ///
"""Public-facing gateway for the local Qwen3.8-Flash-Next server.

The vLLM container stays published on loopback (scripts/serve-public.sh pins it
there); this binds the public interface, enforces bearer auth on /v1/*, and
serves a small dashboard for managing the advertised API URL and the API keys.
Keys live in the gateway, not in the container's argv, so rotating one is
instant -- no restarting a container that loads ~76 GiB of weights.

  uv run scripts/gateway.py --upstream http://127.0.0.1:18300 --port 8080

uv reads the inline dependency block above, so nothing needs installing first.
With aiohttp already on the system, `python3 scripts/gateway.py` works too.

The dashboard is at / and needs the admin token printed at startup.
"""
import argparse, asyncio, hmac, json, os, secrets, socket, sys, time
from pathlib import Path

import aiohttp
from aiohttp import web

# Proxied verbatim (subject to bearer auth). Everything else 404s -- vLLM's
# admin surface (/tokenize, /sleep, ...) has no business being public.
PROXY_PREFIXES = ("/v1/", "/metrics")
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
              "proxy-authorization", "proxy-authenticate", "upgrade", "content-length"}


# ---------------------------------------------------------------- config

class Config:
    """Gateway state, persisted as 0600 JSON at the repo root.

    The file is the source of truth in both directions: the gateway writes
    request counters into it, and an external edit is picked up within a second
    (see `watcher`), so hand-editing a key takes effect without a restart.
    """

    def __init__(self, path: Path):
        self.path = path
        self.stamp = None
        self.warned = None      # stamp of a bad file we have already complained about
        self.data = {"public_url": "", "keys": [], "admin_token": ""}
        disk = self._read()
        if disk:
            self.data.update(disk)
        self._normalize()
        self.save()

    def _read(self):
        """Disk contents, or None if absent or caught mid-write."""
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def _stat(self):
        """Identity of the file as it stands, or None if it is gone."""
        try:
            st = self.path.stat()
            return (st.st_mtime_ns, st.st_size, st.st_ino)  # ino: editors replace, not truncate
        except OSError:
            return None

    def _normalize(self):
        if not self.data.get("admin_token"):
            self.data["admin_token"] = secrets.token_urlsafe(24)
        if not self.data.get("keys"):        # an emptied key list would lock everyone out
            self.data["keys"] = [new_key("default")]
        self.data.setdefault("public_url", "")

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)  # atomic: a crash mid-write can't lose the keys
        self.stamp = self._stat()   # our own write must not read back as an external edit

    def reload_if_changed(self) -> bool:
        """Adopt an external edit, keeping counters. True if anything was reloaded."""
        st = self._stat()
        if st is None or st == self.stamp:
            return False
        disk = self._read()
        if disk is None:
            # Usually an editor mid-write, so don't churn -- but say so once, or a
            # typo'd edit looks indistinguishable from one that simply did nothing.
            if st != self.warned:
                self.warned = st
                print(f"{self.path.name} does not parse as JSON; ignoring it until it does",
                      file=sys.stderr, flush=True)
            return False
        # The edit is the intent, so identity fields come from disk. Counters stay
        # ours: whoever opened the editor was looking at a pre-traffic snapshot.
        mine = {k["id"]: k for k in self.data["keys"] if "id" in k}
        self.data = {"public_url": "", "keys": [], "admin_token": ""}
        self.data.update(disk)
        for k in self.data["keys"]:
            prev = mine.get(k.get("id"))
            if prev and prev.get("requests", 0) > k.get("requests", 0):
                k["requests"], k["last_used"] = prev["requests"], prev["last_used"]
        self._normalize()
        # Write the merged result straight back. Don't try to detect "did anything
        # change?" first: update() aliases disk's own lists into self.data, so the
        # merge above mutates `disk` too and any comparison against it reads equal.
        self.save()
        return True

    def match(self, presented: str):
        """Constant-time lookup of a presented bearer token."""
        for k in self.data["keys"]:
            if hmac.compare_digest(k["key"], presented):
                return k
        return None


def new_key(label: str) -> dict:
    return {"id": secrets.token_hex(6), "label": label, "key": "sk-" + secrets.token_urlsafe(32),
            "created": time.time(), "last_used": None, "requests": 0}


def lan_ip() -> str:
    """Address of the interface that reaches the default route (no traffic sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ---------------------------------------------------------------- auth

def bearer(request) -> str:
    auth = request.headers.get("Authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def admin_ok(request) -> bool:
    cfg = request.app["cfg"]
    tok = (request.headers.get("X-Admin-Token")
           or request.query.get("token")
           or request.cookies.get("admin_token") or "")
    return bool(tok) and hmac.compare_digest(tok, cfg.data["admin_token"])


def deny(msg, status=401):
    # OpenAI-shaped so client SDKs surface something intelligible.
    return web.json_response({"error": {"message": msg, "type": "invalid_request_error",
                                        "code": "invalid_api_key"}}, status=status)


# ---------------------------------------------------------------- proxy

async def proxy(request):
    cfg = request.app["cfg"]
    if not any(request.path.startswith(p) for p in PROXY_PREFIXES):
        return deny("not found", 404)
    key = cfg.match(bearer(request))
    if key is None:
        return deny("Incorrect API key provided.")

    key["requests"] += 1
    key["last_used"] = time.time()
    request.app["state"]["dirty"] = True

    url = request.app["upstream"] + request.rel_url.path_qs
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    headers.pop("Host", None)
    # The client credential is ours, not the upstream's: terminate it here and
    # present the upstream's own key (if vLLM was started with --api-key).
    headers.pop("Authorization", None)
    if request.app["upstream_key"]:
        headers["Authorization"] = "Bearer " + request.app["upstream_key"]
    try:
        up = await request.app["state"]["client"].request(
            request.method, url, headers=headers,
            data=request.content if request.body_exists else None,
            allow_redirects=False)
    except aiohttp.ClientError as e:
        return deny(f"upstream unreachable: {e}", 502)

    async with up:
        out = web.StreamResponse(status=up.status, headers={
            k: v for k, v in up.headers.items() if k.lower() not in HOP_BY_HOP})
        await out.prepare(request)
        try:
            async for chunk in up.content.iter_any():   # iter_any: don't stall SSE
                await out.write(chunk)
        except (ConnectionResetError, asyncio.CancelledError):
            pass                                        # client hung up mid-stream
        await out.write_eof()
        return out


# ---------------------------------------------------------------- admin API

async def upstream_status(app):
    hdrs = {"Authorization": "Bearer " + app["upstream_key"]} if app["upstream_key"] else {}
    try:
        async with app["state"]["client"].get(app["upstream"] + "/v1/models", headers=hdrs,
                                     timeout=aiohttp.ClientTimeout(total=3)) as r:
            if r.status != 200:
                return {"up": False, "detail": f"HTTP {r.status}"}
            body = await r.json()
            return {"up": True, "models": [m["id"] for m in body.get("data", [])]}
    except Exception as e:
        return {"up": False, "detail": str(e).split("\n")[0][:120]}


async def api_state(request):
    cfg = request.app["cfg"]
    return web.json_response({
        "public_url": cfg.data["public_url"],
        "upstream": request.app["upstream"],
        "keys": cfg.data["keys"],
        "server": await upstream_status(request.app),
    })


async def api_config(request):
    cfg = request.app["cfg"]
    body = await request.json()
    if "public_url" in body:
        cfg.data["public_url"] = body["public_url"].strip().rstrip("/")
    cfg.save()
    return web.json_response({"ok": True, "public_url": cfg.data["public_url"]})


async def api_keys(request):
    cfg = request.app["cfg"]
    body = await request.json() if request.body_exists else {}
    cfg.data["keys"].append(new_key(body.get("label", "").strip() or "key"))
    cfg.save()
    return web.json_response({"ok": True, "keys": cfg.data["keys"]})


async def api_key_op(request):
    cfg = request.app["cfg"]
    kid, op = request.match_info["kid"], request.match_info["op"]
    keys = cfg.data["keys"]
    idx = next((i for i, k in enumerate(keys) if k["id"] == kid), None)
    if idx is None:
        return deny("no such key", 404)
    if op == "revoke":
        if len(keys) == 1:
            return deny("cannot revoke the last key", 409)
        keys.pop(idx)
    elif op == "rotate":
        keys[idx].update(key="sk-" + secrets.token_urlsafe(32), requests=0,
                         last_used=None, created=time.time())
    elif op == "label":
        keys[idx]["label"] = (await request.json()).get("label", "").strip() or "key"
    else:
        return deny("unknown op", 400)
    cfg.save()
    return web.json_response({"ok": True, "keys": keys})


@web.middleware
async def admin_guard(request, handler):
    if request.path == "/" or request.path.startswith("/admin/"):
        if not admin_ok(request):
            return web.Response(status=401, content_type="text/html", text=UNAUTHORIZED_HTML)
        resp = await handler(request)
        tok = request.query.get("token")
        if tok:  # promote ?token= to a cookie so the URL can be trimmed
            resp.set_cookie("admin_token", tok, httponly=True, samesite="Lax", max_age=30 * 86400)
        return resp
    return await handler(request)


async def dashboard(request):
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def healthz(request):
    return web.json_response({"ok": True})   # unauthenticated liveness for probes


# ---------------------------------------------------------------- html

UNAUTHORIZED_HTML = """<!doctype html><meta charset=utf-8>
<title>401</title><style>body{font:15px system-ui;margin:4rem auto;max-width:34rem;color:#333}
code{background:#eee;padding:.15em .35em;border-radius:4px}</style>
<h2>Admin token required</h2><p>Open the dashboard with <code>/?token=YOUR_ADMIN_TOKEN</code>.
The token is printed when the gateway starts, and stored in <code>gateway.json</code>.</p>"""

DASHBOARD_HTML = r"""<!doctype html>
<meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Qwen3.8-Flash-Next gateway</title>
<style>
:root{--bg:#faf9f7;--fg:#1c1a17;--mut:#6b6560;--card:#fff;--line:#e5e0d8;--acc:#b4541f}
@media(prefers-color-scheme:dark){:root{--bg:#16150f;--fg:#eae6df;--mut:#9a938a;--card:#211f18;--line:#332f26;--acc:#e08a4c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:52rem;margin:0 auto;padding:2rem 1.25rem 4rem}
h1{font-size:1.35rem;margin:0 0 .2rem}
.sub{color:var(--mut);font-size:.9rem;margin-bottom:1.75rem}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1.1rem 1.25rem;margin-bottom:1.1rem}
h2{font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin:0 0 .85rem;font-weight:600}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
input{font:inherit;padding:.5rem .6rem;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--fg);flex:1;min-width:12rem}
button{font:inherit;font-size:.87rem;padding:.45rem .8rem;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--acc);color:var(--acc)}
button.p{background:var(--acc);border-color:var(--acc);color:#fff}
button.p:hover{opacity:.88;color:#fff}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em}
.dot{width:.55rem;height:.55rem;border-radius:50%;display:inline-block;margin-right:.45rem}
.up{background:#3f9d5a}.down{background:#c4453c}
table{width:100%;border-collapse:collapse}
td,th{padding:.55rem .4rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);font-weight:600}
tr:last-child td{border-bottom:0}
.k{font-family:ui-monospace,monospace;font-size:.8rem;word-break:break-all}
.mask{color:var(--mut)}
.meta{color:var(--mut);font-size:.78rem;white-space:nowrap}
.acts{text-align:right;white-space:nowrap}
.acts button{padding:.28rem .55rem;font-size:.78rem;margin-left:.3rem}
pre{background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:.8rem;overflow-x:auto;margin:0;font-size:.82rem}
.hint{color:var(--mut);font-size:.8rem;margin-top:.55rem}
</style>
<main>
<h1>Qwen3.8-Flash-Next gateway</h1>
<div class=sub>API keys and the advertised endpoint for the local model server.</div>

<section>
  <h2>Server</h2>
  <div id=status class=row><span class=dot></span><span>checking…</span></div>
</section>

<section>
  <h2>API base URL</h2>
  <div class=row>
    <input id=url placeholder="http://host:8080/v1">
    <button class=p onclick=saveUrl()>Save</button>
    <button onclick="copy(document.getElementById('url').value)">Copy</button>
  </div>
  <div class=hint>What clients should point at. Override it if you front the gateway with a
  tunnel or domain; it is only advertised here, never enforced.</div>
</section>

<section>
  <h2>API keys</h2>
  <table><thead><tr><th>Label</th><th>Key</th><th>Used</th><th></th></tr></thead>
  <tbody id=keys></tbody></table>
  <div class=row style="margin-top:.9rem">
    <input id=label placeholder="label for a new key (e.g. laptop)">
    <button class=p onclick=addKey()>New key</button>
  </div>
</section>

<section>
  <h2>Quick start</h2>
  <pre id=snippet></pre>
</section>
</main>
<script>
let S = null, shown = {};
const api = (p, o) => fetch('/admin' + p, Object.assign({headers:{'Content-Type':'application/json'}}, o)).then(r => r.json());
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ago = t => { if(!t) return 'never'; const d = Date.now()/1000 - t;
  if(d<60) return 'just now'; if(d<3600) return Math.floor(d/60)+'m ago';
  if(d<86400) return Math.floor(d/3600)+'h ago'; return Math.floor(d/86400)+'d ago'; };

async function load(){ S = await api('/state'); render(); }

function render(){
  const st = document.getElementById('status');
  st.innerHTML = S.server.up
    ? `<span class="dot up"></span><span>up — <code>${esc((S.server.models||[]).join(', ')||'no models')}</code>
       <span class=meta>&nbsp;via ${esc(S.upstream)}</span></span>`
    : `<span class="dot down"></span><span>down <span class=meta>${esc(S.server.detail||'')} — start it with scripts/serve.sh</span></span>`;

  const u = document.getElementById('url');
  if (document.activeElement !== u) u.value = S.public_url;

  document.getElementById('keys').innerHTML = S.keys.map(k => `<tr>
    <td>${esc(k.label)}</td>
    <td class=k>${shown[k.id] ? esc(k.key) : `<span class=mask>${esc(k.key.slice(0,7))}…${esc(k.key.slice(-4))}</span>`}</td>
    <td class=meta>${k.requests} req · ${ago(k.last_used)}</td>
    <td class=acts>
      <button onclick="toggle('${k.id}')">${shown[k.id] ? 'Hide' : 'Show'}</button>
      <button onclick="copy('${k.key}')">Copy</button>
      <button onclick="op('${k.id}','rotate','Rotate this key? Clients using it stop working immediately.')">Rotate</button>
      <button onclick="op('${k.id}','revoke','Revoke this key permanently?')">Revoke</button>
    </td></tr>`).join('');

  const base = S.public_url || location.origin + '/v1';
  const k = S.keys[0] ? S.keys[0].key : 'YOUR_KEY';
  const model = (S.server.models || ['MODEL'])[0];
  document.getElementById('snippet').textContent =
`curl ${base}/chat/completions \\
  -H "Authorization: Bearer ${k}" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"${model}","messages":[{"role":"user","content":"hi"}]}'

# OpenAI SDK
client = OpenAI(base_url="${base}", api_key="${k}")`;
}

const toggle = id => { shown[id] = !shown[id]; render(); };
const copy = t => navigator.clipboard.writeText(t);
const saveUrl = () => api('/config', {method:'POST', body: JSON.stringify({public_url: document.getElementById('url').value})}).then(load);
const addKey = () => { const el = document.getElementById('label');
  api('/keys', {method:'POST', body: JSON.stringify({label: el.value})}).then(() => { el.value=''; load(); }); };
const op = (id, o, msg) => { if (!confirm(msg)) return;
  api(`/keys/${id}/${o}`, {method:'POST', body:'{}'}).then(r => { if(r.error) alert(r.error.message); load(); }); };

load(); setInterval(load, 5000);
</script>
"""


# ---------------------------------------------------------------- wiring

async def watcher(app):
    """Pick up external edits to the config, and persist counters periodically.

    Both halves live in one loop so a reload and a counter flush can never race
    each other into writing conflicting copies of the file.
    """
    cfg, state = app["cfg"], app["state"]
    tick = 0
    try:
        while True:
            await asyncio.sleep(1)
            tick += 1
            try:
                if cfg.reload_if_changed():
                    state["dirty"] = False   # disk is authoritative again
                    print(f"gateway.json reloaded — {len(cfg.data['keys'])} key(s)",
                          file=sys.stderr, flush=True)
            except Exception as e:           # a bad edit must not kill the gateway
                print(f"gateway.json reload failed, keeping current config: {e}",
                      file=sys.stderr, flush=True)
            if tick % 10 == 0 and state["dirty"]:
                cfg.save()
                state["dirty"] = False
    except asyncio.CancelledError:
        if state["dirty"]:
            cfg.save()
        raise


async def on_start(app):
    app["state"]["client"] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))
    app["state"]["watcher"] = asyncio.create_task(watcher(app))


async def on_stop(app):
    app["state"]["watcher"].cancel()
    try:
        await app["state"]["watcher"]
    except asyncio.CancelledError:
        pass
    await app["state"]["client"].close()


def build(cfg, upstream, upstream_key=""):
    app = web.Application(middlewares=[admin_guard], client_max_size=64 * 1024 * 1024)
    app["cfg"], app["upstream"], app["upstream_key"] = cfg, upstream.rstrip("/"), upstream_key
    # aiohttp deprecates mutating app[...] once it has started, so everything that
    # changes at runtime lives inside this one dict, created before startup.
    app["state"] = {"dirty": False, "client": None, "watcher": None}
    app.router.add_get("/", dashboard)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/admin/state", api_state)
    app.router.add_post("/admin/config", api_config)
    app.router.add_post("/admin/keys", api_keys)
    app.router.add_post("/admin/keys/{kid}/{op}", api_key_op)
    app.router.add_route("*", "/{tail:.*}", proxy)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default=os.environ.get("UPSTREAM", "http://127.0.0.1:18300"),
                    help="local vLLM base URL (default %(default)s)")
    ap.add_argument("--host", default=os.environ.get("GW_HOST", "0.0.0.0"),
                    help="interface to bind (default %(default)s -- reachable from the network)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GW_PORT", 8080)))
    ap.add_argument("--upstream-key", default=os.environ.get("UPSTREAM_KEY", ""),
                    help="key to present upstream, if vLLM itself runs with --api-key")
    ap.add_argument("--config",
                    default=str(Path(__file__).resolve().parent.parent / "gateway.json"))
    args = ap.parse_args()

    cfg = Config(Path(args.config))
    if not cfg.data["public_url"]:
        cfg.data["public_url"] = f"http://{lan_ip()}:{args.port}/v1"
        cfg.save()

    admin = f"http://{lan_ip()}:{args.port}/?token={cfg.data['admin_token']}"
    print(f"gateway    {args.host}:{args.port}  ->  {args.upstream}", file=sys.stderr)
    print(f"api base   {cfg.data['public_url']}", file=sys.stderr)
    print(f"dashboard  {admin}", file=sys.stderr)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("note: bound to a public interface -- every /v1 route needs a bearer key, "
              "but nothing here is TLS. Put a tunnel or reverse proxy in front before "
              "exposing it beyond the LAN.", file=sys.stderr)
    web.run_app(build(cfg, args.upstream, args.upstream_key),
                host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
