---
name: visualization-ui
description: Design, build, or extend the Circuit Tracer UI so it honors the product's technical-visualization principles. Use whenever creating or changing interface components, layouts, controls, or views in this app, or when judging whether a UI change matches the product's identity.
---

# Technical Visualization UI

Circuit Tracer's interface is governed by six principles. Every UI change in this project satisfies each one. Where a principle cannot be met, state the conflict and the reason before proceeding; do not let an unmet principle pass silently.

## 1. Visualize the mental model

The interface mirrors how the user thinks about the system, not the code objects or the file layout.

For this app the mental model is the model's data flow:

```text
input → embeddings → attention → residual stream → FFN → output
```

Make navigation resemble the model. The architecture diagram is interactive navigation, never decoration: each stage maps to the internal state it captures, and the diagram is one of the live location controls, not an exhibit.

## 2. One source of truth

A concept owns exactly one canonical state: the selected location, the selected tokens, the scale scope, and the comparison base each exist once.

Every surface reads and writes that same state through the existing manager and renderers. The diagram, navigator, dropdown, activation canvas, and metadata inspector must never disagree — when the same selection appears on several surfaces, it is one state rendered several times. Selection in one control is reflected by the others, never stored apart.

## 3. Visualization first

Give the largest area of the layout to the thing the user is trying to understand: the residual view at the selected location. Controls configure, annotate, or filter that view; they never dominate it. Keep tooling (checkpoint load, export, diagnostics, technical summary, raw metadata) in compact rows and collapsed sections, out of the visualization's space.

## 4. Progressive detail

Show essentials first; disclose the rest behind expansion.

Drill in the user's order: tensor → token → visualization → statistics. Advanced diagnostics, runtime details, comparisons, exports, and raw metadata belong in expandable sections that start collapsed. Never force the full depth of information onto the screen at once.

## 5. Context for every visualization

Never render a heatmap or other chart without enough information to interpret it. Every visible chart carries, when applicable:

- tensor/location name and category
- shape
- selected token(s)
- what the values mean (raw activations, magnitude, delta, ...)
- the scale and its meaning
- min/max or the visible range
- key summary statistics

Hover identity (token position, token text, dimension, raw value) and the title/axis/subtitle disclosures are part of this contract. When a chart changes form or moves surfaces, carry its full context with it.

## 6. Prefer incremental change

When improving the working app:

- preserve working behavior; reuse the existing canonical state and renderers rather than introducing parallel copies
- make the smallest change that satisfies the principle
- implement the highest-value changes first
- validate after each change before moving on

## Conformance

A UI change is done when all six principles are met, or each unmet principle is explicitly waived with a reason. Keep canonical state, renderers, and labels single-source; extend what exists instead of duplicating it.
