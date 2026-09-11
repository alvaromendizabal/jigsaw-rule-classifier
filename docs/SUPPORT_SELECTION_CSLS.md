# CSLS-style support-selection experiment

**Status: implemented and software-tested; no real-data result recorded here yet.**

The completed raw adapted-vector cosine audit changed almost every pair but concentrated heavily on a small number of supports, and the blinded relevance review did not justify GPU inference. This experiment tests a narrower hypothesis: semantic hubness, rather than semantic representations themselves, caused much of that collapse.

We use CSLS-style local scaling, `2*cos(q,s)-r_q-r_s`, separately within each policy's eligible support pool. It keeps the exact frozen adaptation-plan row mapping and still selects one legitimate positive and one legitimate negative example. It never reads query targets, predictions, organizer-released labels, or the consumed holdout.

The first gate is diagnostic only: pair selections must change and maximum support reuse must strictly decline for both classes in every policy. Passing that gate permits only a blinded relevance review. It does **not** authorize GPU inference. Failing it ends the candidate.

This direction is motivated by the observed hub reuse in the previous audit and by the general nearest-neighbor retrieval literature's use of local scaling to reduce hubs. It is intentionally cheaper than another model run and preserves the project's research → hypothesis → implementation → tests → bounded experiment discipline.
