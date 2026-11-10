# Theory Prediction

~~~mermaid
flowchart TD
    A["Earlier edits"] --> B["Induce theory and boundaries<br/>induction/parallel_discovery.py"]
    B --> C["Extract features and freeze rules<br/>features/source_preparation_cli.py"]
    D["Held-out Before text"] --> E["Predict actions from Before<br/>prediction/shared_source_prediction_cli.py"]
    C --> E
    F["Before and After pairs"] --> G["Label observed actions<br/>annotation/action_annotator.py"]
    C --> G
    E --> H["Evaluate and compare cohort rates<br/>evaluation/condition_evolution_cli.py"]
    G --> H
    H -->|Confirmed uncovered cases| B
    H -->|Large prediction-rate gap| I["Review conditions with another model<br/>evaluation/condition_shift_audit_cli.py"]
    I -->|Confirmed coverage gaps| B
~~~
