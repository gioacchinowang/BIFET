# BIFET

**BI**-domain **F**inite **E**lement **T**oolkit

BIFET is a [**deal.ii**](https://www.dealii.org/) based toolkit
for solving partial differential equation(s) living in high dimension space.
Triangulation of a domain with dimension larger than three is not feasible,
which is a technical bottleneck for high-dimension problems.
Particularly, physcists prefers phase-space description for analyzing systems
with micro processes portrayed by macro properties, i.e., fluid dynamics.

BIFET provides an interface for handling a high-dimension domain which consists of
two sub-domains, like spatial and momentum sub-domains in a phase-space domain.
Triangulation in each sub-domain can thus be carried out independently, and as well for other
mathematical quantities like finite-elements and sparsity pattern.
The methods designed for assembling high-dimension system from two sub-domains 
root in deal.ii library.
