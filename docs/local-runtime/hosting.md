# Host a decision model yourself

A tuned decision model is an ordinary directory on your machine:
`.tuned-tensor/decision-runs/<run-id>/train/model`. Tuned Tensor does not
operate a hosting service. To answer requests from another application, run the
model on infrastructure you control: the same machine, a server, a container
platform or a cloud account of your own.

Laya runs on CPU, so no GPU is needed for inference. The former hosted service
ran one Laya checkpoint with 6 GB of CPU memory; use that as a starting point and
measure your own latency and memory before you size anything.

## Load the model

Install the Laya version TT trains with, then load the run's model directory:

```bash
python -m pip install "laya==0.3.22"
```

```python
import laya

agent = laya.load("path/to/decision-runs/<run-id>/train/model")
question = {
    "type": "choice",
    "instructions": "Which support queue should handle this message?",
    "criteria": {
        "lost_card": "A lost or stolen bank card",
        "payment_declined": "A refused card payment",
    },
}
answer = agent.predict("My card payment was declined", {"queue": question})["answers"]["queue"]
print(answer["choice"], answer["probabilities"])
```

Send the same instructions and criteria the model was trained with; they are
part of its input. The run's `resolved-workflow.json` records them under
`question`.

## A minimal HTTP service

The example below uses only the Python standard library besides Laya. It loads
the model once, accepts `POST` requests whose body holds `input` and `question`,
and returns Laya's typed answer.

```python
# serve_decision.py
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import laya

MODEL_DIR = os.environ["TT_DECISION_MODEL"]
API_KEY = os.environ["TT_DECISION_API_KEY"]
MAX_BODY_BYTES = 128 * 1024

agent = laya.load(MODEL_DIR)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {API_KEY}":
            return self.reply(401, {"error": "unauthorized"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            return self.reply(413, {"error": "body too large or empty"})
        try:
            body = json.loads(self.rfile.read(length))
            answer = agent.predict(body["input"], {"answer": body["question"]})["answers"]["answer"]
        except (KeyError, TypeError, ValueError) as error:
            return self.reply(400, {"error": str(error)})
        self.reply(200, {"result": answer})

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
```

```bash
export TT_DECISION_MODEL=.tuned-tensor/decision-runs/<run-id>/train/model
export TT_DECISION_API_KEY="$(openssl rand -hex 32)"
python serve_decision.py
```

```bash
curl http://127.0.0.1:8080 \
  -H "Authorization: Bearer $TT_DECISION_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

`request.json` contains both the input and the question:

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

This is a starting point, not a production server. Before exposing it beyond
your machine:

- Keep the listener on `127.0.0.1` and put a TLS-terminating reverse proxy or
  your platform's load balancer in front of it.
- Store the API key in your platform's secret storage, never in browser code.
- Add the rate limits, logging and health checks your platform expects.
- Keep the model directory read-only and pin `laya==0.3.22` in your image.

## Packaging for your own cloud

Any platform that runs a Python container works. Copy the model directory and
the script into the image, install `laya==0.3.22`, and start
`python serve_decision.py` with the two environment variables set. Bind to
`0.0.0.0` only inside a container whose port is reachable solely through your
proxy or load balancer. You own the account, the costs and the shutdown.
