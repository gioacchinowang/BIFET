# BIFET

**BI**-domain **F**inite **E**lement **T**oolkit

BIFET is a [**deal.ii**](https://www.dealii.org/) based interface
for solving PDEs living in high-dimension (by high-dimension we mean dimension larger than three) space.
Finite element triangulation of a domain with dimension larger than three is not feasible, which is a technical bottleneck for high-dimension problems.
Particularly, physicists prefer phase-space description for analyzing systems with micro processes portrayed by macro properties, e.g., fluid dynamics.
In order to free scientists from the trade-off between simulation precision and programming complexity, a general framework which enables users to describe and solve high-dimension problem is required.

Driven by such motivation, BIFET is designed to decompose high-dimension problems into two sub-domains, e.g., expressing phase-space distribution with spatial and momentum coordinates.
Triangulation in each sub-domain can thus be carried out independently, and as well for other mathematical quantities like finite-elements and sparsity pattern.
The methods invented for assembling high-dimension linear algebra from two sub-domains root deeply in deal.ii library.
