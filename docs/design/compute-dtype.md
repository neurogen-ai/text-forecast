# Compute dtype policy

Promoted from the v2.4.0 compute-dtype proposal (plans/releases/v2.4.0 and
v2.4.1.md). This is the standing rule for any new dtype work.

- Dtypes are explicit end to end. No autocast, no GradScaler anywhere in the
  training loop. Revisit only via a new proposal that asks for fp16
  training in earnest.
- Params and big tensors (db buffer, stage-2 pool gather) live in the
  compute dtype; small, numerics-sensitive tensors (stage-1 top-k score
  matrix, the CLaRa softmax/log-mask loop) stay fp32. The log(mask + 1e-6)
  trick underflows in fp16 and loses too much in bf16.
- fp16 ships with a warning, not an error: the strategy has no GradScaler,
  so fp16 is offered but not recommended.
- Eval builds its context without a dtype (fp32). Checkpoint loads upcast
  silently. If bf16 ever becomes the default working dtype, threading
  `--dtype` through eval is the durable follow-up; the alternative of
  storing dtype in the checkpoint payload grows the format and stays
  rejected until then.
