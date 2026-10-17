const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

function fixture({ failMail = false } = {}) {
  const sent = [];
  const values = new Map();
  const lock = { tryLock: () => true, hasLock: () => true, releaseLock() {} };
  const sandbox = {
    console: { error() {} },
    ContentService: { MimeType: { JSON: "json" }, createTextOutput(text) { return { text, setMimeType() { return this; } }; } },
    LockService: { getScriptLock: () => lock },
    PropertiesService: { getScriptProperties: () => ({
      getProperty: key => values.get(key) || null,
      setProperty: (key, value) => values.set(key, value),
      getProperties: () => Object.fromEntries(values),
      deleteProperty: key => values.delete(key),
    }) },
    Utilities: { newBlob: (text, type, name) => ({ text, type, name }) },
    MailApp: {
      getRemainingDailyQuota: () => 100,
      sendEmail(mail) { if (failMail) throw new Error("offline"); sent.push(mail); },
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(`${__dirname}/Code.gs`, "utf8"), sandbox);
  return { sent, submit(report) {
    const raw = JSON.stringify(report);
    return JSON.parse(sandbox.doPost({ postData: { contents: raw }, contentLength: Buffer.byteLength(raw) }).text);
  } };
}

const report = { schema_version: 1, report_id: "bio-spheres-report-123456-99", version: "test", message: "The menu is lagging", diagnostics: "BIO-SPHERES BUG REPORT\n[perf-menu] fps=3" };

test("message and diagnostics go to the fixed developer mailbox", () => {
  const receiver = fixture();
  assert.equal(receiver.submit(report).ok, true);
  assert.equal(receiver.sent.length, 1);
  assert.equal(receiver.sent[0].to, "Quadraxis77@gmail.com");
  assert.match(receiver.sent[0].body, /The menu is lagging/);
  assert.equal(receiver.sent[0].attachments[0].text, report.diagnostics);
});

test("retrying an accepted ID does not send another email", () => {
  const receiver = fixture();
  receiver.submit(report);
  assert.equal(receiver.submit(report).ok, true);
  assert.equal(receiver.sent.length, 1);
});

test("mail failure never produces a success acknowledgement", () => {
  const receiver = fixture({ failMail: true });
  assert.equal(receiver.submit(report).ok, false);
  assert.equal(receiver.sent.length, 0);
});

test("empty messages and invalid payloads are rejected", () => {
  const receiver = fixture();
  assert.equal(receiver.submit({ ...report, message: " " }).ok, false);
  assert.equal(receiver.submit({ ...report, report_id: "../other" }).ok, false);
  assert.equal(receiver.submit({ ...report, diagnostics: "unrelated" }).ok, false);
  assert.equal(receiver.sent.length, 0);
});

test("a client cannot select another email recipient", () => {
  const receiver = fixture();
  receiver.submit({ ...report, to: "other@example.com" });
  assert.equal(receiver.sent[0].to, "Quadraxis77@gmail.com");
});

test("bursts are queued rather than silently accepted", () => {
  const receiver = fixture();
  for (let i = 0; i < 5; i++) {
    assert.equal(receiver.submit({ ...report, report_id: `bio-spheres-report-123456-${i}` }).ok, true);
  }
  assert.equal(receiver.submit({ ...report, report_id: "bio-spheres-report-123456-6" }).ok, false);
  assert.equal(receiver.sent.length, 5);
});
