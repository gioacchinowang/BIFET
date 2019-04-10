# BIFET

**BI**-domain **F**inite **E**lement **T**oolkit

BIFET is a deal.ii based toolkit
for solving partial differential equation(s) living in high dimension space.
Triangulation of a domain with dimension larger than three is not feasible,
which is a technical bottleneck for high-dimension problems.
Particularly, physcists prefers phase-space description for analyzing systems
with unresolvable micro processes, i.e., fluid dynamics.

BIFET provides a interface for handling a high-dimension domain consists of
two sub-domains, like spatial and momentum sub-domain in phase-space domain.
Triangulation in each sub-domain can thus be carried out independently.
The methods designed for assembling high-dimension system from two sub-domains 
root in deal.ii library.
