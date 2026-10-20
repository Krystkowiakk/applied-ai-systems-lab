"""A live plant-operations mailbox with Gmail-style search.

Three facts hide in ~150 messages. Each has a trap that a careless solver falls into:
the maintenance date is corrected later in its thread (and corrected again while the
solver is working), the VPN code was rotated and a look-alike domain sends a fake one,
and the change ticket has a rejected sibling and an unrelated approved one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.tools import ToolError
from ...core.verdict import Verdict

DOMAIN = "halden.example"
LOOKALIKE = "halden-support.example"
PAGE_SIZE = 10
TODAY = date(2026, 3, 16)
LATE_MAIL_AFTER_CALLS = 3

FIELDS = ("maintenance_date", "vpn_code", "change_id")
FORMATS = {
    "maintenance_date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "vpn_code": re.compile(r"^VX-\d{4}-[A-Z]{2}$"),
    "change_id": re.compile(r"^CHG-\d{6}$"),
}

HELP = """Mail API
search(query, page) -> newest first, 10 per page, metadata and a short snippet only.
  Operators: from:<text>  subject:<word> or subject:"several words"  after:YYYY-MM-DD  before:YYYY-MM-DD
  Any other word must appear in the subject or body (all words must match, case-insensitive).
get_thread(thread_id) -> every message of the thread with full bodies, oldest first.
wait_for_new_mail() -> waits briefly and reports how many messages arrived.
The mailbox is live: messages can arrive while you work."""

FIRST = ["Anna", "Tomas", "Lena", "Marco", "Ines", "Piotr", "Sara", "Jonas", "Elif", "Hugo", "Maja", "Ravi"]
LAST = ["Berg", "Novak", "Rossi", "Weber", "Costa", "Nowak", "Dahl", "Moreau", "Kaya", "Lund", "Silva", "Iyer"]
NOISE = [
    (
        "Shift handover {area}",
        "Night shift handover for {area}: output {n} units, two short stops, scrap within limits. Details in the shift log.",
    ),
    (
        "Supplier delivery {n} delayed",
        "{vendor} reports that delivery {n} for {area} slips by one day because of a truck breakdown. Safety stock covers it.",
    ),
    (
        "Invoice {n} from {vendor}",
        "Please find invoice {n} attached. Payment terms are 30 days. Contact accounts payable if anything is unclear.",
    ),
    (
        "Quality hold on batch {n}",
        "Batch {n} from {area} is on hold pending a dimensional check. Do not ship until QA releases it.",
    ),
    (
        "Safety walk {area} on Thursday",
        "The monthly safety walk in {area} is on Thursday at 10:00. Please make sure walkways are clear.",
    ),
    ("New starter: {person}", "Please welcome {person}, who joins the {area} maintenance team on Monday."),
    (
        "Forklift inspection due",
        "Annual forklift inspection for {area} is due. Book a slot with {vendor} before month end.",
    ),
    (
        "Re: scrap rate {area}",
        "Scrap went up to 3.1% on {area} yesterday. Looks like a tooling issue, checking with the setter.",
    ),
    (
        "Kaizen workshop: changeover times",
        "Join us Friday for a short workshop on reducing changeover times. Coffee provided.",
    ),
    ("Password expiry reminder", "Your directory password expires in 9 days. Change it in the self-service portal."),
    ("Weekly OEE report", "OEE last week: 78% plant average. {area} was lowest because of material shortages."),
    (
        "Spare parts order {n}",
        "Order {n} with {vendor} for bearings and seals is confirmed. Expected in stores next Tuesday.",
    ),
]
AREAS = ["Line 1", "Line 2", "Line 3", "Line 4", "the press shop", "the paint shop", "Hall B", "the finishing area"]
VENDORS = ["Kestrel Bearings", "Brakka Steel", "Orvik Hydraulics", "Norrpak Films", "Vellmar Freight"]


@dataclass
class Mail:
    id: str
    thread_id: str
    sender: str
    subject: str
    sent: datetime
    body: str


def human_date(d: date) -> str:
    return f"{d:%A} {d.day} {d:%B %Y}"


class InboxEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        self._mails: list[Mail] = []
        self._threads = 0
        self._late: Mail | None = None
        self._build()

    # ----- data -----------------------------------------------------------------------
    def _person(self) -> str:
        return f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"

    def _address(self, person: str, domain: str = DOMAIN) -> str:
        first, last = person.lower().split()
        return f"{first}.{last}@{domain}"

    def _when(self, days_ago_min: int, days_ago_max: int) -> datetime:
        day = TODAY - timedelta(days=self.rng.randint(days_ago_min, days_ago_max))
        return datetime(day.year, day.month, day.day, self.rng.randint(7, 18), self.rng.randint(0, 59))

    def _thread(self, subject: str, messages: list[tuple[str, datetime, str]], reply: bool = True) -> str:
        self._threads += 1
        thread_id = f"T{self._threads:04d}"
        for i, (sender, sent, body) in enumerate(sorted(messages, key=lambda m: m[1])):
            prefix = "Re: " if reply and i else ""
            self._mails.append(Mail(f"M{len(self._mails) + 1:05d}", thread_id, sender, prefix + subject, sent, body))
        return thread_id

    def _build(self) -> None:
        r = self.rng
        self.date_announced = TODAY + timedelta(days=r.randint(4, 9))
        self.date_corrected = self.date_announced + timedelta(days=r.choice([1, 2, 7]))
        self.date_final = self.date_corrected + timedelta(days=r.choice([2, 3, 5]))
        it_ops = f"it-ops@{DOMAIN}"
        announce = self._when(9, 12)
        correction = self._when(2, 4)
        self.maintenance_thread = self._thread(
            "Planned maintenance: manufacturing execution system (MES)",
            [
                (
                    it_ops,
                    announce,
                    f"Hello all, the MES will be offline on {human_date(self.date_announced)} from 22:00 to 02:00 for the database upgrade. Lines keep running; terminals will queue bookings locally.",
                ),
                (
                    it_ops,
                    correction,
                    f"Correction: because of the quarter-end freeze the MES maintenance window moves to {human_date(self.date_corrected)}, same hours. Apologies for the churn.",
                ),
            ],
        )
        self._late = Mail(
            "M_LATE",
            self.maintenance_thread,
            it_ops,
            "Re: Planned maintenance: manufacturing execution system (MES)",
            datetime(TODAY.year, TODAY.month, TODAY.day, 9, 41),
            f"Final update: the database vendor can only support us on {human_date(self.date_final)}. The MES maintenance window is now {human_date(self.date_final)}, 22:00 to 02:00. This replaces all earlier dates.",
        )

        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        self.vpn_old = f"VX-{r.randint(1000, 9999)}-{r.choice(letters)}{r.choice(letters)}"
        self.vpn_current = f"VX-{r.randint(1000, 9999)}-{r.choice(letters)}{r.choice(letters)}"
        self.vpn_phish = f"VX-{r.randint(1000, 9999)}-{r.choice(letters)}{r.choice(letters)}"
        security = f"security@{DOMAIN}"
        self._thread(
            "Contractor VPN access",
            [
                (
                    security,
                    self._when(30, 40),
                    f"Maintenance contractor VPN access code for this quarter: {self.vpn_old}. Do not forward this message.",
                )
            ],
        )
        self._thread(
            "Contractor VPN access codes rotated",
            [
                (
                    security,
                    self._when(1, 3),
                    f"We rotated contractor credentials today. New maintenance contractor VPN access code: {self.vpn_current}. The previous code stops working at midnight.",
                )
            ],
        )
        self._thread(
            "URGENT: verify your VPN access code",
            [
                (
                    f"security@{LOOKALIKE}",
                    self._when(0, 1),
                    f"Our records show your contractor VPN access needs re-validation. Use code {self.vpn_phish} and confirm at the link below within 24 hours.",
                )
            ],
            reply=False,
        )

        self.change_approved = f"CHG-{r.randint(400000, 499999)}"
        self.change_rejected = f"CHG-{int(self.change_approved[4:]) - r.randint(1, 9)}"
        self.change_unrelated = f"CHG-{r.randint(300000, 399999)}"
        board = f"change-board@{DOMAIN}"
        self._thread(
            "CAB decisions this week",
            [
                (
                    board,
                    self._when(6, 8),
                    f"CAB outcome: {self.change_rejected} (MES maintenance, first draft) was rejected: rollback plan missing.",
                ),
                (
                    board,
                    self._when(3, 5),
                    f"CAB outcome: {self.change_approved} (MES maintenance) is approved. Implementation follows the window announced by IT operations.",
                ),
            ],
        )
        self._thread(
            "CAB decision: printer fleet",
            [
                (
                    board,
                    self._when(10, 14),
                    f"CAB outcome: {self.change_unrelated} (office printer fleet replacement) is approved.",
                )
            ],
        )

        for _ in range(110):
            subject, body = r.choice(NOISE)
            values = {
                "n": r.randint(10000, 99999),
                "area": r.choice(AREAS),
                "vendor": r.choice(VENDORS),
                "person": self._person(),
            }
            sender = self._address(self._person())
            first = (sender, self._when(0, 45), body.format(**values))
            messages = [first]
            if r.random() < 0.25:
                messages.append(
                    (
                        self._address(self._person()),
                        first[1] + timedelta(hours=r.randint(1, 30)),
                        "Thanks, noted. I will follow up.",
                    )
                )
            self._thread(subject.format(**values), messages)

    # ----- API ------------------------------------------------------------------------
    def _tick(self) -> None:
        if self._late is not None and self.api_calls >= LATE_MAIL_AFTER_CALLS:
            self._mails.append(self._late)
            self._late = None

    def help(self) -> str:
        self._api("help")
        self._tick()
        return HELP

    def search(self, query: str, page: int = 1) -> dict[str, Any]:
        self._api("search")
        self._tick()
        if len(query) > 200:
            raise ToolError("Query too long (max 200 characters).")
        matches = [m for m in self._mails if _matches(m, query)]
        matches.sort(key=lambda m: m.sent, reverse=True)
        pages = max(1, -(-len(matches) // PAGE_SIZE))
        if page < 1 or page > pages:
            raise ToolError(f"Page {page} out of range.", hint=f"There are {pages} page(s).")
        chunk = matches[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        result: dict[str, Any] = {
            "total": len(matches),
            "page": page,
            "pages": pages,
            "results": [
                {
                    "message_id": m.id,
                    "thread_id": m.thread_id,
                    "from": m.sender,
                    "subject": m.subject,
                    "date": m.sent.isoformat(timespec="minutes"),
                    "snippet": m.body[:70],
                }
                for m in chunk
            ],
        }
        if not matches:
            result["hint"] = "No messages matched. Drop an operator or use fewer, more general words."
        elif page < pages:
            result["hint"] = f"More results on page {page + 1}."
        return result

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        self._api("get_thread")
        self._tick()
        messages = sorted((m for m in self._mails if m.thread_id == thread_id), key=lambda m: m.sent)
        if not messages:
            raise ToolError(f"No thread {thread_id}.", hint="Use thread_id values returned by search.")
        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "message_id": m.id,
                    "from": m.sender,
                    "date": m.sent.isoformat(timespec="minutes"),
                    "subject": m.subject,
                    "body": m.body,
                }
                for m in messages
            ],
        }

    def wait_for_new_mail(self) -> dict[str, int]:
        self._api("wait_for_new_mail")
        arrived = 0
        if self._late is not None:
            self._mails.append(self._late)
            self._late = None
            arrived = 1
        return {"new_messages": arrived}

    def submit(self, answer: dict[str, str]) -> Verdict:
        self._api("submit")
        self._tick()
        truth = {
            "maintenance_date": (self.date_final if self._late is None else self.date_corrected).isoformat(),
            "vpn_code": self.vpn_current,
            "change_id": self.change_approved,
        }
        stale = {
            "maintenance_date": {self.date_announced.isoformat(), self.date_corrected.isoformat()}
            - {truth["maintenance_date"]},
            "vpn_code": {self.vpn_old},
            "change_id": {self.change_rejected},
        }
        status: dict[str, str] = {}
        for field in FIELDS:
            value = str(answer.get(field, "")).strip()
            if not value:
                status[field] = "missing"
            elif not FORMATS[field].match(value):
                status[field] = "bad format"
            elif value == truth[field]:
                status[field] = "ok"
            elif value in stale[field]:
                status[field] = "outdated: a newer message replaced this value"
            elif field == "vpn_code" and value == self.vpn_phish:
                status[field] = "rejected: this code came from an untrusted sender"
            else:
                status[field] = "wrong"
        accepted = all(s == "ok" for s in status.values())
        feedback = "; ".join(f"{f}: {s}" for f, s in status.items())
        return self._record(Verdict(accepted, "OK" if accepted else "FIELDS_REJECTED", feedback, {"fields": status}))


_OPERATOR = re.compile(r'(from|subject|after|before):("([^"]*)"|\S+)|"([^"]+)"|(\S+)', re.IGNORECASE)


def _matches(mail: Mail, query: str) -> bool:
    text = f"{mail.subject}\n{mail.body}".lower()
    for op, raw, quoted, phrase, word in _OPERATOR.findall(query):
        value = (quoted if raw.startswith('"') else raw) if op else (phrase or word)
        value = value.lower()
        op = op.lower()
        if op == "from" and value not in mail.sender.lower():
            return False
        if op == "subject" and value not in mail.subject.lower():
            return False
        if op in ("after", "before"):
            try:
                bound = date.fromisoformat(value)
            except ValueError as exc:
                raise ToolError(f"Bad date in {op}: '{value}'. Use YYYY-MM-DD.") from exc
            if (op == "after" and mail.sent.date() <= bound) or (op == "before" and mail.sent.date() >= bound):
                return False
        if not op and value not in text:
            return False
    return True
