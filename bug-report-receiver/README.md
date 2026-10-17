# Bug report receiver — one-time developer setup

Player flow: **F8 → describe the bug → Send report**. The game uploads the
message and diagnostics automatically. Players need no email app, browser,
account, attachment selection, or other setup.

The receiver is deployed. Its public URL is stored in `endpoint.txt` and is
embedded automatically by `cargo build --release`; no environment variable is
needed for normal builds. Email arrives under Gmail's **Bio-Spheres Bug Reports**
label and skips the main inbox. Reports remain unread until opened.

Live delivery has been verified with a synthetic message and diagnostics
attachment. No report is marked sent without a matching acknowledgement from
the receiver. Older executables need to be rebuilt to include this URL.

## Updating or replacing the receiver

1. Open [Google Apps Script](https://script.google.com/) using the Google account
   that should send reports to `Quadraxis77@gmail.com`, and create a project.
2. Paste `Code.gs` into the project. In Project Settings, enable showing the
   manifest and replace `appsscript.json` with the file here.
3. Choose **Deploy → New deployment → Web app**. Set **Execute as: Me** and
   **Who has access: Anyone**. Authorize sending mail when Google asks. Access
   must allow anonymous requests, so players never need to sign in.
4. Copy the deployed URL ending in `/exec`.
5. Replace the URL in `endpoint.txt` and rebuild. To temporarily override the
   receiver when building the game (PowerShell):

   ```powershell
   $env:BIO_SPHERES_BUG_REPORT_URL = 'https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec'
   cargo build --release
   ```

The URL is included in the executable. Players do not set an environment
variable. No Gmail password, OAuth token, or private mail credential goes into
the game. Update the deployment to a new version after changing the script.

The receiver sends only to the fixed developer address, validates report size
and schema, deduplicates retries by report ID, and limits new emails to five per
minute. It confirms acceptance only after `MailApp.sendEmail` succeeds. A
timeout or rejected request remains queued on the player's machine and retries
automatically, including after restarting the game.

Consumer Google accounts currently allow **100 email recipients per day**, shared
with other MailApp use. This receiver is suitable for a small alpha; reports
remain queued if the quota is reached. See Google's
[mail quotas](https://developers.google.com/apps-script/guides/services/quotas)
and [web app deployment instructions](https://developers.google.com/apps-script/guides/web).

Run local receiver tests with `node --test bug-report-receiver/receiver.test.cjs`.
They mock MailApp and do not send email.

To explicitly send a synthetic report through the game's HTTPS client:

```powershell
cargo test --lib live_receiver_accepts_a_synthetic_report -- --ignored --nocapture
```

This sends one email with a short test attachment and no player logs or settings.
The live test is ignored during ordinary test runs.

## Gmail filing

The Gmail filter matches:

```text
from:quadraxis77@gmail.com to:quadraxis77@gmail.com subject:"Bio-Spheres bug report"
```

It applies **Bio-Spheres Bug Reports** and **Skip the Inbox (Archive it)**. It
does not mark reports read, delete them, or forward them to another address.
If the sending account changes, update this filter's sender address too.

## Collected data and privacy

| Data | Diagnostic purpose |
| --- | --- |
| CPU model, logical CPU count, OS version/architecture, total and available RAM | Hardware compatibility and memory pressure |
| GPU model, driver, backend, enabled features/limits | Driver problems and rendering fallbacks |
| Build version/profile, source fingerprint, build time, VR feature | Distinguish builds sharing the same version number |
| Window/render resolution, scaling, refresh rate, focus, presentation mode/frame cap | Frame pacing and display configuration |
| OpenXR runtime/headset model, eye/UI resolution, session state and refresh | Desktop versus VR presentation problems |
| F8 snapshot and Send snapshot | Keep the original incident while the player writes a description |
| Recent frame times and p95/p99, 120 bounded performance intervals, game/system CPU separately, sample ages and memory | Diagnose intermittent lag and stale counters |
| Named GPU timings with sample age, menu CPU phases, cached cell counts, physics settings, sun/cave/render state | Separate expensive work and reproduce relevant settings |
| Filtered current/previous-session performance records, warning/error source locations and counts, last panic source location/build | Problems that disappear after restarting |

Hardware models and software versions are retained; hardware serials, persistent
device/user IDs, process IDs, monitor/audio-device names, IP/MAC addresses,
account identifiers, usernames, screenshots, genome/mode names and contents,
VR tracking/controller poses, arbitrary launch arguments and environment dumps
are not collected as diagnostic fields. Report IDs are random for each report.
No other processes or unrelated files are enumerated.

Raw log messages are excluded. Only `[perf]`, `[perf-cpu]`, `[perf-menu]` and
`[diagnostics-build]` records are retained, with log timestamp prefixes removed.
The six whitelisted settings files are parsed, and string values/comments are
excluded. Parse failures and oversized files are explicitly marked instead of
falling back to raw content. Panic messages/backtraces are excluded; only the
source filename, line/column and build identity are retained. Current in-memory
settings are included because saved settings can lag behind the incident.
Warning/error sites are counted separately without recording their message text.
Full world saves and genome definitions are not attached, so reports provide
diagnostic context rather than a complete replay of the simulation.

Local redaction removes known home paths/usernames/hostnames, absolute local and
network paths, email addresses, IP/MAC addresses, UUIDs, common phone-number
formats, URLs and credential-like values. It runs before saving a new report,
when loading queued reports, and immediately before HTTPS upload. Older queued
reports have their original diagnostics omitted and their message scrubbed
before delivery. The diagnostic attachment is capped at 1 MiB; sections and the
performance history have independent limits. Hardware queries and file reads
run on the report worker, not the render thread.

**This is identifier minimization and redaction, not a guarantee of complete
anonymity.** Free-form descriptions can contain names or identifiers the
scrubber cannot recognize, and unusual hardware combinations can be identifying.
The dialog asks players to omit personal details. Google necessarily receives
connection metadata when handling HTTPS requests; that metadata is not included
in the report attachment. The receiver sends reports only to the developer's
mailbox, where they remain until deleted by the developer. A player who needs
absolute anonymity should not submit a report through this flow.
