// Serves the s1bench R2 bucket at /<READ_KEY>/<object key>. GET and HEAD honour Range (including suffix ranges) and answer
// 206 with Content-Range, as an object store does. Writes (PUT, and multipart: POST ?uploads, PUT ?uploadId&partNumber,
// POST ?uploadId) also need the x-s1-write header to equal WRITE_KEY. Both keys are Worker secrets.
export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const parts = url.pathname.split("/");
    if (parts.length < 3 || parts[1] !== env.READ_KEY) return new Response("not found", { status: 404 });
    const key = parts.slice(2).map(decodeURIComponent).join("/");
    if (!key) return new Response("no key", { status: 400 });
    const m = req.method;
    if (m === "GET" || m === "HEAD") return read(env, key, req, m === "HEAD");
    if (req.headers.get("x-s1-write") !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
    const q = url.searchParams;
    if (m === "PUT" && !q.has("uploadId")) {
      await env.BUCKET.put(key, req.body);
      return Response.json({ ok: true });
    }
    if (m === "POST" && q.has("uploads")) {
      const mpu = await env.BUCKET.createMultipartUpload(key);
      return Response.json({ uploadId: mpu.uploadId });
    }
    if (m === "PUT" && q.has("uploadId")) {
      const mpu = env.BUCKET.resumeMultipartUpload(key, q.get("uploadId"));
      const part = await mpu.uploadPart(Number(q.get("partNumber")), req.body);
      return Response.json({ partNumber: part.partNumber, etag: part.etag });
    }
    if (m === "POST" && q.has("uploadId")) {
      const mpu = env.BUCKET.resumeMultipartUpload(key, q.get("uploadId"));
      await mpu.complete(await req.json());
      return Response.json({ ok: true });
    }
    return new Response("method not allowed", { status: 405 });
  },
};

async function read(env, key, req, head) {
  if (head) {
    const o = await env.BUCKET.head(key);
    if (!o) return new Response(null, { status: 404 });
    return new Response(null, { headers: { "Content-Length": String(o.size), ETag: o.httpEtag, "Accept-Ranges": "bytes" } });
  }
  // R2 echoes the range option back, so the Range header is parsed here and the served slice computed from it
  const r = parseRange(req.headers.get("range"));
  const o = await env.BUCKET.get(key, r ? { range: r } : {});
  if (!o) return new Response("not found", { status: 404 });
  const h = new Headers({ ETag: o.httpEtag, "Accept-Ranges": "bytes" });
  if (!r) {
    h.set("Content-Length", String(o.size));
    return new Response(o.body, { headers: h });
  }
  let off, len;
  if (r.suffix !== undefined) {
    len = Math.min(r.suffix, o.size);
    off = o.size - len;
  } else {
    off = r.offset;
    len = Math.min(r.length ?? o.size - off, o.size - off);
  }
  h.set("Content-Range", `bytes ${off}-${off + len - 1}/${o.size}`);
  h.set("Content-Length", String(len));
  return new Response(o.body, { status: 206, headers: h });
}

function parseRange(v) {
  const m = v && /^bytes=(\d*)-(\d*)$/.exec(v.trim());
  if (!m || (m[1] === "" && m[2] === "")) return null;
  if (m[1] === "") return { suffix: Number(m[2]) };
  const offset = Number(m[1]);
  return m[2] === "" ? { offset } : { offset, length: Number(m[2]) - offset + 1 };
}
