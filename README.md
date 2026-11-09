# Theory Prediction

A Python workflow for deriving editing rules from before-and-after text, predicting editing actions on held-out text, and comparing observed action rates across cohorts.

## Workflow

1. **Derive a theory from the earlier cohort.** Open-code editing actions and candidate conditions, group comparable actions, and find supporting and contrasting evidence. Synthesize rules jointly with their boundaries. The parallel discovery and hierarchical synthesis interfaces live in `theory_discovery.parallel_discovery` and `theory_discovery.hierarchical_theory`. The condition review in `theory_discovery.annotation_abstraction` folds retained limits into the finalized conditions.
2. **Extract and freeze the features.** `theory_discovery.source_preparation_cli` prepares shared condition features, action definitions, and Boolean rules from typed source bundles. Conditions and actions have separate definitions.
3. **Predict on the held-out cohort.** `theory_discovery.shared_source_prediction_cli` measures conditions from Before text and applies the frozen rules to produce action labels. It saves a prediction snapshot before outcomes are joined.
4. **Label observed actions independently.** The annotation stage accepts before-and-after pairs and the frozen action protocol. The model cites numbered changes, and the program retrieves and validates the corresponding text on both sides. Validated labels are saved per sample for resumption. The earlier cohort can be prepared as a statistics baseline without predicting its actions.
5. **Evaluate and compare.** `theory_discovery.condition_evolution_cli` joins frozen predictions to observed action labels and compares observed rates under each condition. Each cohort must have at least five usable samples under that condition before a percentage-point difference is reported. `theory_discovery.uncovered_discovery_cli` can route confirmed out-of-scope cases to a separate induction run.

## Running from source

Use Python 3.11 or newer. Install `transformers` to run tokenizer-based preflights or model-backed stages. A compatible model server and its local tokenizer are supplied by the operator. Set `SIMULATION_MODEL_BASE_URL` to the OpenAI-compatible endpoint and `SIMULATION_TOKENIZER_PATH` to the tokenizer directory. The compact model-backed commands expect the server to advertise `qwen3.8-27b`.

From the repository root, these commands inspect the local configuration and CLI contracts:

```sh
python -m theory_discovery.run_pipeline
python -m theory_discovery.source_preparation_cli --help
python -m theory_discovery.shared_source_prediction_cli --help
python -m theory_discovery.condition_evolution_cli --help
python -m theory_discovery.uncovered_discovery_cli --help
```

The bundled configuration disables data access and model execution. Induction input uses JSON Lines with `before` and `after` strings. Prediction and evaluation accept the typed source, Before-only, outcome, and annotation records defined in `theory_discovery.stage_interfaces` and `theory_discovery.workflow_contracts`.

With prepared labels, the rate comparison can run entirely offline:

```sh
python -m theory_discovery.condition_evolution_cli \
  --old-records old_labels.json \
  --new-records new_labels.json \
  --output work/rates
```

To build an earlier-cohort statistics baseline as part of that command, replace `--old-records` with `--prepare-old-baseline baseline_config.json --action-protocol action_protocol.json`. The baseline configuration contains `compiled_sources`, `before`, `outcomes`, and `workspace` paths. Use a dedicated workspace for checkpoints and a new directory for the comparison output. Baseline preparation explicitly uses the configured model service.

Keep datasets, generated outputs, caches, and credentials in local storage outside version control.
