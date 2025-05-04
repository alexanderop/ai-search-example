#!/usr/bin/env node

import fs from "fs/promises";
import path from "path";
import { glob } from "glob";
import matter from "gray-matter";
import { remark } from "remark";
import strip from "strip-markdown";
import ProgressBar from "cli-progress";
import { pipeline, cos_sim, env } from "@xenova/transformers";

// ---------- Config ----------
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";      // fast, 384-dim embeddings
const OUTPUT_DIR = "./public";
const EMBEDDING_FILE = path.join(OUTPUT_DIR, "semantic-index.json");
const CONTENT_DIR = "src/data/blog";
const MODELS_DIR_PUBLIC = path.join(OUTPUT_DIR, "models");
const BATCH_SIZE = 16;                              // texts per forward pass
const MAX_CONCURRENCY = 2;                          // batches in parallel
// --------------------------------

// ---------- Helpers ----------
async function copyDir(src, dest) {
  await fs.mkdir(dest, { recursive: true });
  for (const entry of await fs.readdir(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      await copyDir(srcPath, destPath);
    } else {
      await fs.copyFile(srcPath, destPath);
    }
  }
}

function deriveSlug(filePath, fmSlug) {
  if (fmSlug && fmSlug.trim()) {
    return fmSlug.trim();
  }
  let slug = path
    .relative(CONTENT_DIR, filePath)
    .replace(/\\/g, "/")
    .replace(/\.mdx?$/, "")
    .replace(/\/index$/, "");
  return slug.toLowerCase();
}

async function markdownToPlain(md) {
  const file = await remark().use(strip).process(md);
  return String(file).replace(/\s+/g, " ").trim();
}

// paragraph-level chunks; returns [{id,text}]
function chunkText(text, baseSlug) {
  const paras = text.split(/\n{2,}/g).filter((p) => p.trim().length > 40);
  return paras.map((p, i) => ({
    id: `${baseSlug}#para-${i}`,
    text: p,
  }));
}
// --------------------------------

// ---------- Bootstrap ----------
(async () => {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(MODELS_DIR_PUBLIC, { recursive: true });

  const extractor = await pipeline("feature-extraction", MODEL_NAME, { quantized: true });

  // Copy model to public for client-side
  try {
    const cachePath = path.join(env.cacheDir, MODEL_NAME);
    const publicPath = path.join(MODELS_DIR_PUBLIC, MODEL_NAME);
    await fs.access(publicPath).catch(async () => {
      await copyDir(cachePath, publicPath);
    });
  } catch (e) {
  }

  const files = await glob(`${CONTENT_DIR}/**/*.md*`, {
    ignore: [`${CONTENT_DIR}/**/_*.md*`],
  });
  if (!files.length) {
    await fs.writeFile(EMBEDDING_FILE, "[]");
    return;
  }

  // Load previous index for incremental builds
  let prev = {};
  try {
    const raw = JSON.parse(await fs.readFile(EMBEDDING_FILE, "utf-8"));
    raw.forEach((e) => { prev[e.id] = e.mtime; });
  } catch {}

  const embeddings = [];
  const bar = new ProgressBar.SingleBar({ hideCursor: true });
  bar.start(files.length, 0);

  let queue = [];

  async function flushBatch() {
    if (!queue.length) return;
    const batch = queue.splice(0, BATCH_SIZE);
    const texts = batch.map((b) => b.text);
    const outputs = await extractor(texts, { pooling: "mean", normalize: true });
    batch.forEach((item, i) => {
      embeddings.push({
        ...item.meta,
        vector: Array.from(outputs[i].data),
      });
      bar.increment();
    });
  }

  for (const filePath of files) {
    const stat = await fs.stat(filePath);
    const mtime = +stat.mtime;
    const contentRaw = await fs.readFile(filePath, "utf-8");
    const { content, data: fm } = matter(contentRaw);

    if (fm.draft) {
      bar.increment();
      continue;
    }

    const slug = deriveSlug(filePath, fm.slug);
    const title = fm.title || slug;
    const description = fm.description || "";

    // Skip unchanged
    if (prev[slug] && prev[slug] === mtime) {
      bar.increment();
      continue;
    }

    const plain = await markdownToPlain(content);
    const chunks = chunkText(plain, slug);

    chunks.forEach((chunk) => {
      queue.push({
        meta: { id: chunk.id, slug, title, description, mtime },
        text: `${title}. ${description}. ${chunk.text}`,
      });
    });

    // throttle
    if (queue.length >= BATCH_SIZE * MAX_CONCURRENCY) {
      await flushBatch();
    }
  }

  // flush remainder
  await flushBatch();
  bar.stop();

  await fs.writeFile(EMBEDDING_FILE, JSON.stringify(embeddings));

  if (embeddings.length >= 2) {
    const sim = cos_sim(embeddings[0].vector, embeddings[1].vector);
  }
})();
