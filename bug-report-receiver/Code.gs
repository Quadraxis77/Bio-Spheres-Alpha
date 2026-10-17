// Deploy as a web app: execute as Me; access Anyone (including anonymous users).
// Players never sign in. This receiver can send only to this fixed mailbox.
const REPORT_TO = "Quadraxis77@gmail.com";
const MAX_REQUEST_BYTES = 8 * 1024 * 1024;
const RECEIPT_TTL_MS = 7 * 24 * 60 * 60 * 1000;

function reply(value) {
  return ContentService.createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}

function doPost(event) {
  let lock;
  try {
    const raw = event && event.postData && event.postData.contents;
    if (typeof raw !== "string" || raw.length > MAX_REQUEST_BYTES ||
        event.contentLength > MAX_REQUEST_BYTES) {
      return reply({ ok: false, error: "Invalid report size" });
    }
    const report = JSON.parse(raw);
    if (report.schema_version !== 1 ||
        typeof report.report_id !== "string" ||
        !/^bio-spheres-report-[0-9]+-[0-9]+$/.test(report.report_id) ||
        report.report_id.length > 100 ||
        typeof report.message !== "string" || !report.message.trim() || report.message.length > 8000 ||
        typeof report.diagnostics !== "string" || !report.diagnostics.startsWith("BIO-SPHERES BUG REPORT") ||
        report.diagnostics.length > 2 * 1024 * 1024 ||
        typeof report.version !== "string" || report.version.length > 64) {
      return reply({ ok: false, error: "Invalid report" });
    }

    lock = LockService.getScriptLock();
    if (!lock.tryLock(5000)) return reply({ ok: false, error: "Receiver busy" });
    const properties = PropertiesService.getScriptProperties();
    const key = "receipt:" + report.report_id;
    if (properties.getProperty(key)) return reply({ ok: true, report_id: report.report_id });

    const now = Date.now();
    const minute = Math.floor(now / 60000);
    const recent = JSON.parse(properties.getProperty("recent") || "{}");
    if (recent.minute === minute && recent.count >= 5) {
      return reply({ ok: false, error: "Receiver busy" });
    }
    if (MailApp.getRemainingDailyQuota() < 1) {
      return reply({ ok: false, error: "Daily mail quota reached" });
    }
    MailApp.sendEmail({
      to: REPORT_TO,
      subject: "Bio-Spheres bug report — " + report.report_id,
      body: report.message.trim() + "\n\nGame version: " + report.version +
        "\nReport ID: " + report.report_id + "\n\nDiagnostics are attached.",
      attachments: [Utilities.newBlob(report.diagnostics, "text/plain", report.report_id + ".txt")],
    });
    // Acknowledge only after the mail service accepts the message and attachment.
    properties.setProperty(key, String(now));
    properties.setProperty("recent", JSON.stringify({ minute, count: recent.minute === minute ? recent.count + 1 : 1 }));
    const saved = properties.getProperties();
    for (const name of Object.keys(saved)) {
      if (name.startsWith("receipt:") && now - Number(saved[name]) > RECEIPT_TTL_MS) {
        properties.deleteProperty(name);
      }
    }
    return reply({ ok: true, report_id: report.report_id });
  } catch (error) {
    console.error("Bug report delivery failed", String(error));
    return reply({ ok: false, error: "Report not accepted; retry later" });
  } finally {
    if (lock && lock.hasLock()) lock.releaseLock();
  }
}
