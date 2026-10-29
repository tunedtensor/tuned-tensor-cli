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
# Use a completed decision run's checkpoint and its recorded question.
tt cloud endpoints create \
  --model .tuned-tensor/decision-runs/<run-id>/train/model \
  --question .tuned-tensor/decision-runs/<run-id>/resolved-workflow.json \
  --name support-triage
```

The command uploads only `model.safetensors` (at most 2 GB), without loading it
into a model runtime. The serving image supplies the trusted Laya architecture
and tokenizer. You do not need to upload the base model as well. To host the base
model instead, point `--model` at its local checkpoint directory and use the same
question. `--question` also accepts a JSON file containing just the Laya question:
`type`, `instructions` and `criteria`.

Save the endpoint ID, HTTPS inference URL and inference key returned by create.
The key is shown only once. It can invoke only this endpoint; your account token
continues to manage it. Check status until it reports `ready`:

```bash
tt cloud endpoints get <endpoint-id>
tt cloud endpoints list
tt cloud endpoints invoke <endpoint-id> --input "My card payment was declined"
```

Each successful request costs the fixed number of US cents displayed at creation.
The price stays fixed for that endpoint. Failed requests are refunded. This
feature does not debit an idle compute or deployment fee; the operator still
incurs AWS storage and related costs. Use `tt balance` and `tt balance --help`
for your account balance and billing commands. Each account may have five
non-deleted endpoints and create at most ten endpoints in 24 hours.

A request has an idempotency UUID, printed by `invoke`. If a connection drops or
the API says the request is still running, retry the **same input** with that ID:

```bash
tt cloud endpoints invoke <endpoint-id> --input "My card payment was declined" \
  --request-id <original-request-uuid>
```

Replays return the recorded answer without a second charge. Reusing the ID with
different input is rejected. A failed/refunded request remains failed; use a new
ID for a fresh attempt. Interrupted requests are refunded by periodic recovery
within roughly 15 minutes. Input is limited to 8192 characters; one request
answers the question pinned at endpoint creation.

For another application, call the URL from its backend:

```bash
curl "$TT_INFERENCE_URL" \
  -H "Authorization: Bearer $TT_INFERENCE_KEY" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  -d '{"input":"My card payment was declined"}'
```

The response's `data.result` is Laya's typed answer with probabilities. Store keys
in your backend's secret storage; do not put them in public browser code. The
service stores results for idempotent replay and input hashes rather than input
text. Endpoint keys can be rotated at any time:

```bash
tt cloud endpoints rotate-key <endpoint-id>
tt cloud endpoints delete <endpoint-id>
tt cloud endpoints get <endpoint-id>
```

Deletion immediately blocks new requests. Cleanup removes the model, weights
and SageMaker resources asynchronously; a newly created endpoint may stay
`deleting` for up to 16 minutes while its upload URL expires. Poll until
`deleted`. Uploads abandoned for an hour, deployments that cannot become ready
within an hour, and failed endpoints older than a day are cleaned automatically.
Account deletion requires endpoint cleanup and accepted requests to finish.

If creation fails after returning an endpoint ID, inspect that ID or delete it.
If a deployment exists but the one-time key was lost, rotate it. Do not repeat
`create` simply to retrieve the key: that allocates another endpoint.
