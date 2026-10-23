# Guardrails: what stands between untrusted text, personal data and a model

Code: [`src/aisyslab/guardrails/`](../src/aisyslab/guardrails/). Lab:
[`quality-claims`](labs.md#quality-claims).

A customer complaint is untrusted text full of personal data. The guardrails here are code around
the model calls, not instructions to the model: each one holds whether or not the model cooperates.

## Classify, then anonymise in layers

`Anonymizer.classify` sorts a text before anything is sent: **red** for payment data (an IBAN, a
card number that passes the Luhn check), **yellow** for personal data (names, e-mail addresses,
phone numbers), **green** otherwise. `Anonymizer.anonymize` replaces personal data with
placeholders such as `[PERSON_1]`, in layers: patterns first (exact and cheap), then a directory of
known people (a CRM), then, optionally, a model for what both missed. The mapping stays in code:
`Redacted.restore` puts the real values back into a reply written with placeholders, after every
model has seen it.

Red data never goes to a cloud model. With `--local-model`, a red e-mail is read by the model on
this machine; without one, the red values are removed before any call.

## Moderation, with thresholds per category

`OpenAIModeration` calls OpenAI's moderation model (free of charge when this was written) and
flags the categories whose score passes their threshold: lower for threats than for rudeness,
because a missed threat costs more. `KeywordModeration` is the offline stand-in. Both implement
`flagged`, so either plugs into the service gateway's moderation hook. A flagged message goes to a
person; no model writes back to it.

## Quarantined extraction

`extract` gives the untrusted text to a model that has no tools and must answer with JSON matching
a schema of enums, numbers and identifiers; an answer that does not validate is sent back once
with the errors. The agent that can act never sees the message, only the validated fields, so an
instruction in the message ("approve a full refund") has nothing to act through.

## A topic guard on replies

`TopicGuard` has a second model check every reply against a policy (no amounts of money, no
admission of liability, no other topics, no instructions followed from the customer) before it is
sent. The writer and the checker are separate calls with separate instructions.

## A signed, chained audit log

`AuditLog` signs every entry with an HMAC and chains it to the previous one, so a changed, removed
or reordered entry fails `verify`. Model calls are logged as hashes and lengths, not text: the log
proves what happened without becoming a second copy of the personal data.

## The leak check

The lab's environment watches the boundary: every prompt bound for a cloud model passes
`check_outgoing`, which blocks it if it contains a name, address, phone number or account from the
claims. The check caught a gap while this was being built: phone numbers with a one-digit group
(`+31 6 1234 5678`) slipped past the anonymiser's pattern.

## A live run

On 24 October 2026, `gpt-5-mini` at low effort with OpenAI's moderation handled seed 0 in one
submission, for about a cent: the threatening e-mail was flagged (harassment 0.57) and escalated
without a reply, the e-mail with a bank account was classified red and handled without the
account reaching a model, and the e-mail addressed to "the AI assistant" got an ordinary reply
with no amount in it.
