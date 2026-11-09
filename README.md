# Theory Prediction

~~~mermaid
flowchart TD
    A["Earlier edits"] --> B["Induce theory and boundaries<br/>induction/"]
    B --> C["Extract features and freeze rules<br/>features/"]
    C --> E["Predict actions from Before text<br/>prediction/"]
    D["Held-out Before text"] --> E
    F["Before and After pairs"] --> G["Label observed actions<br/>annotation/"]
    C --> G
    E --> H["Evaluate predictions and compare cohorts<br/>evaluation/"]
    G --> H
    H -->|Confirmed uncovered cases| B
~~~
