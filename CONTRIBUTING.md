# Contributing to vMAR

vMAR is a research-stage C23 library. **External contributions are not yet
being solicited or accepted into a public release**. This file describes
the intended review contract; it does not authorize redistribution of the
current source archive while ownership/license review is incomplete.

Future contributions should:

1. Explain the mathematical capability and its relation to DSNeuralRNAS.
2. Provide an explicit API contract (domain, dimensions, memory ownership,
   numeric failure behavior, state/attempt semantics, and evaluation cost).
3. Add unit and invariant tests; use independent R references for claims
   of R–C23 numerical equivalence.
4. Keep experimental controls opt-in and preserve M0/M1 semantics.
5. Document whether any external code or data was adapted, its copyright
   owner, original URL/revision, license and notices.
6. Update technical documentation in English and the Spanish user entry
   points when exposed behavior changes.
7. Avoid claiming a superior algorithm or a proved stability property
   solely from small empirical studies.

Before accepting external contributions, the maintainer must establish a
clear rights and contribution policy compatible with the final license.
