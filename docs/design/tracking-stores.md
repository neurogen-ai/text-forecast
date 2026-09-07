# Tracking stores: the gather boundary and the feed rule

Settled in the v2.3.1 process (plans/proposals/v2.3.1-mlflow-logging-fix.md;
the first ratification's release doc was withdrawn for amendment and
re-ratified the same day; these decisions stand throughout).
These rules outlive the patch
and constrain the 2.4.0 compute-dtype plumbing, which feeds bf16 tensors
into the same stores.

1. Every tracker metric, plot, and export read crosses
   `MetricTracker._gather_store`. Nothing outside metric_tracker.py touches
   `tracker.stores` directly, so dtype conversion needed at the numpy
   boundary belongs there, in one place. It does not belong at individual
   `.numpy()` call sites or in strategy feed code: numpy awareness is the
   tracker's job, and the strategy feeds what the model produced.
2. Stores are fed; gathers are never made conditional on emptiness. An empty
   store after a guard is an expected condition and logs once per store
   name, not per epoch. Id stores are symmetric: the train and val paths
   feed the same store names, and a NaN-fallback id batch (from
   `return_id=False` configs) skips the feed instead of feeding junk.
