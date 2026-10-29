# Host a decision model

Use a serverless endpoint when a small decision model receives occasional
requests and you want to keep it off your own machine. Hosting currently supports
the pinned English Laya base checkpoint and complete TT decision fine-tunes.
GPU models, adapters, custom tokenizers and custom calibration are not supported.
The endpoint uses 6 GB of CPU memory and scales to zero when idle; the first
request after an idle period may be slow or fail while it starts.

You need a TT account token and prepaid credits. Your application does not need
AWS credentials. The hosted API must be enabled by the service operator before
these commands work.

```bash
tt auth login
tt balance
# Use a completed decision run's checkpoint.
tt cloud endpoints create \
  --model .tuned-tensor/decision-runs/<run-id>/train/model \
  --name support-triage
```

The command uploads only `model.safetensors` (at most 2 GB), without loading it
into a model runtime. The serving image supplies the trusted Laya architecture
and tokenizer. You do not need to upload the base model as well. To host the base
model instead, point `--model` at its local checkpoint directory. There is no
question or other inference configuration attached to a deployment. Send the
complete model request each time you invoke it.

Save the endpoint ID, HTTPS inference URL and inference key returned by create.
The key is shown only once. It can invoke only this endpoint; your account token
continues to manage it. Check status until it reports `ready`:

```bash
tt cloud endpoints get <endpoint-id>
tt cloud endpoints list
tt cloud endpoints invoke <endpoint-id> --payload request.json
```

For the current Laya runtime, `request.json` contains both the input and question:

```json
{
  "input": "My card payment was declined",
  "question": {
    "type": "choice",
    "instructions": "Which support queue should handle this message?",
    "criteria": {
      "lost_card": "A lost or stolen bank card",
      "payment_declined": "A refused card payment"
    }
  }
}
```

You may send a different question on the next request. The CLI sends this JSON
object as-is; model-specific fields belong in the request, not deployment options.
The hosting API bounds requests to 128 KiB; Laya validates its own fields.

Each successful request costs the fixed number of US cents displayed at creation.
The price stays fixed for that endpoint. Failed requests are refunded. This
feature does not debit an idle compute or deployment fee; the operator still
incurs AWS storage and related costs. Use `tt balance` and `tt balance --help`
for your account balance and billing commands. Each account may have five
non-deleted endpoints and create at most ten endpoints in 24 hours.

A request has an idempotency UUID, printed by `invoke`. If a connection drops or
the API says the request is still running, retry the **same complete payload** with that ID:

```bash
tt cloud endpoints invoke <endpoint-id> --payload request.json \
  --request-id <original-request-uuid>
```

Replays return the recorded answer without a second charge. Reusing the ID with
a different payload (including question or options) is rejected. A failed/refunded request remains failed; use a new
ID for a fresh attempt. Interrupted requests are refunded by periodic recovery
within roughly 15 minutes. For Laya, input text is limited to 8192 characters; each request
includes the question it should answer. After three failures in five
minutes, wait five minutes before a new attempt. After 100 failures across the
account in 24 hours, new requests pause until earlier failures age out; cached
replays remain available.

For another application, call the URL from its backend:

```bash
curl "$TT_INFERENCE_URL" \
  -H "Authorization: Bearer $TT_INFERENCE_KEY" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

The response's `data.result` is Laya's typed answer with probabilities. Store keys
in your backend's secret storage; do not put them in public browser code. The
service stores results for idempotent replay and request hashes rather than the request
body. You can also open **Dashboard → Endpoints**, choose your endpoint, and click
**Generate new inference key**. Confirm replacement, then copy the new key shown
once. The old key stops working immediately; update your backend's secret.
This page also lets you copy the inference URL and refresh deployment status.
Settings continues to manage account access tokens, which have broader access.

Endpoint keys can also be rotated from the CLI:

```bash
tt cloud endpoints rotate-key <endpoint-id>
tt cloud endpoints delete <endpoint-id>
tt cloud endpoints get <endpoint-id>
```

Deletion immediately blocks new requests. Cleanup removes the model, weights
and SageMaker resources asynchronously; an endpoint still provisioning must finish
that transition before AWS permits deletion. Late uploads go to temporary storage
with automatic expiry and cannot recreate the serving artifact. Poll until
`deleted`. Uploads abandoned for an hour, deployments that cannot become ready
within an hour, and failed endpoints older than a day are cleaned automatically.
Account deletion requires endpoint cleanup and accepted requests to finish.

If creation fails after returning an endpoint ID, inspect that ID or delete it.
If a deployment exists but the one-time key was lost, rotate it. Do not repeat
`create` simply to retrieve the key: that allocates another endpoint.
