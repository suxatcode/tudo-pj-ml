#import "./lib.typ": TODO

= Final Report: Machine Learning for Physicists

== Model Description
As unsupervised loss function the paper (see equation (1) in @paper-graphsage)
uses $
  J_G(z_u) = -log(sigma(z_u^T z_v)) - Q * EE_(v_n~P_n) log(sigma(-z_u^T z_(v_n))),
$
where $z_u, u in VV$ are the node embeddings (output), $v$ is a node that
co-occurs near $u$ on a fixed length random walk, $sigma$ is the sigmoid
function, $P_n$ is a negative sampling distribution and $Q$ defines the number
of negative samples.

#TODO[Define our loss function(s)]

#bibliography("references.bib")
