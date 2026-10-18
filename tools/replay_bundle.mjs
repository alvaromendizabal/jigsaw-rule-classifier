/* Public replay transport shared with the same owner's Kaggriculture project.
 * Domain validation lives in replay_public_demo.mjs; no private artifacts enter this bundle. */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";

export const requireThat = (condition, message) => {
  if (!condition) throw new Error(message);
};
export const digest = (value) =>
  crypto.createHash("sha256").update(value).digest("hex");
const json = (value) => JSON.stringify(value, null, 2) + "\n";
const SCOPE =
  "EDITABLE_LOCAL_TEXT: current public lexical engine replay; no private model or historical score reproduction.";
const AUTHENTICATION =
  "Local consistency and replay evidence; hashes are not an independent signature or historical attestation.";
const files = ["export.json", "receipt.json", "manifest.json"];
const sources = [
  "public-demo/engine.js",
  "public-demo/app.js",
  "tools/replay_bundle.mjs",
  "tools/replay_public_demo.mjs",
];
export function safePath(filename) {
  const resolved = path.resolve(filename);
  let current = path.parse(resolved).root;
  for (const part of resolved.slice(current.length).split(path.sep)) {
    current = path.join(current, part);
    try {
      requireThat(
        !fs.lstatSync(current).isSymbolicLink(),
        "Symlink paths are not supported.",
      );
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  return resolved;
}
export function readBytes(filename) {
  const target = safePath(filename);
  const fd = fs.openSync(
    target,
    fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK,
  );
  try {
    const stat = fs.fstatSync(fd);
    requireThat(
      stat.isFile() && stat.size <= 8 * 1024 * 1024,
      "Expected a regular JSON file of at most 8 MiB.",
    );
    const bytes = Buffer.alloc(stat.size + 1);
    let length = 0;
    while (length < bytes.length) {
      const count = fs.readSync(fd, bytes, length, bytes.length - length, null);
      if (count === 0) break;
      length += count;
    }
    requireThat(length === stat.size, "File changed size while reading.");
    return bytes.subarray(0, length);
  } finally {
    fs.closeSync(fd);
  }
}
export function readJson(filename) {
  return JSON.parse(readBytes(filename).toString("utf8"));
}
function validateReceipt(receipt, project) {
  match(
    [
      "schema",
      "project",
      "status",
      "scope",
      "created_utc",
      "runtime",
      "elapsed_seconds",
      "source_sha256",
      "verified",
      "authentication",
    ].sort(),
    Object.keys(receipt).sort(),
    "receipt.fields",
  );
  match(
    {
      schema: "public-replay-receipt-v1",
      project,
      status: "PASS",
      scope: SCOPE,
      authentication: AUTHENTICATION,
    },
    Object.fromEntries(
      ["schema", "project", "status", "scope", "authentication"].map((key) => [
        key,
        receipt[key],
      ]),
    ),
    "receipt.contract",
  );
  requireThat(
    typeof receipt.created_utc === "string" &&
      Number.isFinite(Date.parse(receipt.created_utc)) &&
      new Date(receipt.created_utc).toISOString() === receipt.created_utc,
    "Invalid receipt UTC time.",
  );
  requireThat(
    typeof receipt.elapsed_seconds === "number" &&
      Number.isFinite(receipt.elapsed_seconds) &&
      receipt.elapsed_seconds >= 0,
    "Invalid receipt elapsed time.",
  );
  requireThat(
    receipt.runtime && typeof receipt.runtime === "object",
    "Invalid runtime metadata.",
  );
  match(
    ["node", "platform", "arch", "cpus"].sort(),
    Object.keys(receipt.runtime).sort(),
    "runtime.fields",
  );
  requireThat(
    typeof receipt.runtime.node === "string" &&
      /^v[0-9]+\.[0-9]+\.[0-9]+$/.test(receipt.runtime.node),
    "Invalid Node runtime.",
  );
  for (const key of ["platform", "arch"])
    requireThat(
      typeof receipt.runtime[key] === "string" &&
        /^[a-z0-9_]+$/.test(receipt.runtime[key]),
      "Invalid runtime identity.",
    );
  requireThat(
    Number.isInteger(receipt.runtime.cpus) && receipt.runtime.cpus > 0,
    "Invalid CPU count.",
  );
}

export function match(expected, actual, location = "$", tolerance = 0) {
  if (typeof expected === "number") {
    requireThat(
      typeof actual === "number" && Number.isFinite(actual),
      `Nonfinite number at ${location}`,
    );
    const bound = Number.isInteger(expected)
      ? 0
      : tolerance * (1 + Math.abs(expected));
    requireThat(
      Math.abs(expected - actual) <= bound,
      `Replay mismatch at ${location}`,
    );
  } else if (expected === null || typeof expected !== "object") {
    requireThat(expected === actual, `Replay mismatch at ${location}`);
  } else {
    requireThat(
      actual !== null &&
        typeof actual === "object" &&
        Array.isArray(expected) === Array.isArray(actual),
      `Invalid shape at ${location}`,
    );
    const left = Object.keys(expected).sort(),
      right = Object.keys(actual).sort();
    requireThat(
      JSON.stringify(left) === JSON.stringify(right),
      `Unexpected or missing fields at ${location}`,
    );
    for (const key of left)
      match(expected[key], actual[key], `${location}.${key}`, tolerance);
  }
}
function sourceHashes(root) {
  return Object.fromEntries(
    sources.map((name) => [
      name,
      digest(fs.readFileSync(safePath(path.join(root, name)))),
    ]),
  );
}
function outsideCheckout(root, output) {
  const relative = path.relative(root, output);
  requireThat(
    relative.startsWith(".." + path.sep) || path.isAbsolute(relative),
    "Use an output directory outside the checkout.",
  );
}
export function runBundle(
  { root, project, generate, replay },
  argv = process.argv.slice(2),
) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    requireThat(
      ["--output", "--input", "--check"].includes(key) &&
        !Object.hasOwn(args, key),
      "Use --output DIR [--input EXPORT.json] [--check].",
    );
    if (key === "--check") args[key] = true;
    else {
      requireThat(
        typeof argv[i + 1] === "string" && !argv[i + 1].startsWith("--"),
        `Missing value for ${key}`,
      );
      args[key] = argv[++i];
    }
  }
  requireThat(
    typeof args["--output"] === "string",
    "Use --output DIR [--input EXPORT.json] [--check].",
  );
  const output = safePath(args["--output"]);
  outsideCheckout(root, output);
  const currentSources = sourceHashes(root);
  if (fs.existsSync(output)) {
    requireThat(
      fs.statSync(output).isDirectory(),
      "Output is not a directory.",
    );
    match(files.slice().sort(), fs.readdirSync(output).sort(), "bundle.files");
    const manifest = readJson(path.join(output, "manifest.json"));
    match(
      {
        schema: "public-replay-manifest-v1",
        files: Object.fromEntries(
          ["export.json", "receipt.json"].map((name) => [
            name,
            digest(readBytes(path.join(output, name))),
          ]),
        ),
      },
      manifest,
      "manifest",
    );
    const receipt = readJson(path.join(output, "receipt.json"));
    validateReceipt(receipt, project);
    match(currentSources, receipt.source_sha256, "source_sha256");
    requireThat(
      receipt.schema === "public-replay-receipt-v1" &&
        receipt.project === project &&
        receipt.status === "PASS",
      "Invalid replay receipt.",
    );
    const exported = readJson(path.join(output, "export.json"));
    if (args["--input"]) match(readJson(args["--input"]), exported, "input");
    match(replay(exported), receipt.verified, "verified");
    console.log(
      JSON.stringify({
        status: "PASS",
        mode: "replayed",
        output,
        verified: receipt.verified,
      }),
    );
    return receipt;
  }
  requireThat(!args["--check"], "No existing replay bundle to verify.");
  const exported = args["--input"] ? readJson(args["--input"]) : generate();
  const started = performance.now(),
    verified = replay(exported);
  const receipt = {
    schema: "public-replay-receipt-v1",
    project,
    status: "PASS",
    scope: SCOPE,
    created_utc: new Date().toISOString(),
    runtime: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      cpus: os.availableParallelism(),
    },
    elapsed_seconds: (performance.now() - started) / 1000,
    source_sha256: currentSources,
    verified,
    authentication: AUTHENTICATION,
  };
  const parent = path.dirname(output);
  requireThat(
    fs.statSync(parent).isDirectory(),
    "Create the parent directory first.",
  );
  const stage = fs.mkdtempSync(path.join(parent, ".public-replay-"));
  try {
    fs.writeFileSync(path.join(stage, "export.json"), json(exported), {
      flag: "wx",
    });
    fs.writeFileSync(path.join(stage, "receipt.json"), json(receipt), {
      flag: "wx",
    });
    fs.writeFileSync(
      path.join(stage, "manifest.json"),
      json({
        schema: "public-replay-manifest-v1",
        files: Object.fromEntries(
          ["export.json", "receipt.json"].map((name) => [
            name,
            digest(fs.readFileSync(path.join(stage, name))),
          ]),
        ),
      }),
      { flag: "wx" },
    );
    requireThat(
      !fs.existsSync(output),
      "Output appeared during replay; refusing replacement.",
    );
    fs.renameSync(stage, output);
  } finally {
    fs.rmSync(stage, { recursive: true, force: true });
  }
  console.log(
    JSON.stringify({ status: "PASS", mode: "created", output, verified }),
  );
  return receipt;
}
