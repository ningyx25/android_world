# AndroidWorld Agents

## Overview

The agents in this folder are designed to interact with Android devices by
perceiving the device state and taking appropriate actions to accomplish user
goals. The framework uses a hierarchical structure with a base agent class that
specialized agents extend.

## Agents

### M3A (Multimodal Autonomous Agent for Android)

- Featured in the AndroidWorld paper (ICLR 2025)
- Uses both visual and textual data to interact with Android devices
- Capable of understanding screenshots and text descriptions of UI elements
- Uses Set-of-Mark action space

### T3A (Text-only Autonomous Agent for Android)

- Featured in the AndroidWorld paper (ICLR 2025)
- Text-only version of M3A
- Uses only textual representations of UI elements without visual information
- Works with the same action space as M3A

### SeeAct

- Featured in the AndroidWorld paper (ICLR 2025)
- Web-adapted version of the agent from "GPT-4V(ision) is a Generalist Web Agent, if Grounded"
- Uses visual grounding with a two-step reasoning process
- Specialized for interacting with Android interfaces

### Mobile Jev (`mobile_jev.py`, `mobile_jev_v2.py`)

- Decision-model agents rather than prompt agents: the code owns candidate
  discovery, validation, coordinate resolution and execution, and the model
  answers structured questions about one screen. Both run against the Docker
  server through `run_on_docker.py` (`--agent_name=client_mobile_jev` and
  `client_mobile_jev_v2`).
- `ClientMobileJev` (v1) is text-only: the UI tree reaches the model as
  `state.elements`, and `WAIT`, `DONE` and `BLOCKED` are operations it may
  choose.
- `ClientMobileJevV2` is the agent half of the Dohnuts `ac-jev-v2` recipe, so it
  is bound to the rows that recipe trained: the model sees screenshots (the
  frame, its set-of-mark rendering, and the frames of the last five actions),
  `state` carries only `goal` and `recentActions`, the four scroll operations
  collapse into one `SCROLL` whose direction a second question chooses, and
  completion is a separate yes/no question rather than a `DONE` operation. It
  also takes a multimodal LLM, which names the text a `TYPE_TEXT` types: the
  one decision the checkpoint has no trained readout for.
- The module docstrings carry the full v1-to-v2 list and the reasons.

#### The decision request v2 sends

`infer.TypeSafeJevWrapper.predict_jev_mm` posts `model`, `state`, `questions`
and `images` to `TYPESAFE_MM_ENDPOINT`. That is the body the kev sidecar
defines (`kev.api.SystemOneRequest`) and its path is the TypeSafe one, so a
local server and the hosted model are interchangeable behind it:

    uv run --extra serve python -m kev.serve --run <vision-run> --port 8008
    export TYPESAFE_MM_ENDPOINT=http://localhost:8008/v1/systemone

`images` is a bare list of base64 PNGs and the order is part of the contract:
history oldest first, then the raw frame being decided on, then its set-of-mark
rendering when the screen offers TAP candidates. Nothing in that list names a
frame, so the client applies the pixel budget each kind was trained on (512^2
for history, 1024^2 for the current frames) before sending it.

A server that wants no key, as a local one is with KEV_API_KEY unset,
is reached by exporting TYPESAFE_MM_ENDPOINT alone; the key stays required
while the hosted default endpoint is in play.

The server answers every question id it was given, and the agent consumes only
the branch the operation question selected: a `choice` answer carries
`choice`, `probabilities` and `confidence`; a `noul` answer carries `noul`.

Two questions the training data has no rows for are built but never sent:
`scroll_target`, because the corpus records a scroll direction but never the
region that moved, and `text_value`, because no row was converted for a typed
value. The scrolled region follows the convention that labels a history scroll
in training (the lowest-indexed scroll candidate of the screen), and the text to
type is chosen from the exact spans of the goal by the multimodal LLM. A kind the
screen offers only one candidate of gets no question, and that candidate is
taken as resolved.

Because a checkpoint with no vision backbone drops the frames and answers from
the state alone, the wrapper reads the serving card at the sibling
`/v1/models` path on its first vision call and says out loud when that
endpoint does not look like a vision one.


### Base and Utility Classes

- `base_agent.py`: Abstract base class defining the agent interface
- `agent_utils.py`: Common utility functions used by agents
- `infer.py`: Inference utilities for working with language models
- `*_utils.py`: Agent-specific utility functions

## Usage

Agents implement the `EnvironmentInteractingAgent` interface, providing a
consistent way to:

- Get the device state
- Take actions on the device
- Process feedback after actions
- Track progress through a series of interactions

Each agent has specific initialization requirements but follows the same basic
interaction pattern through the `step()` method.