# PMIC: Improving Multi-Agent Reinforcement Learning with Progressive Mutual Information Collaboration
Original PyTorch implementation of PMIC from PMIC: Improving Multi-Agent Reinforcement Learning with Progressive Mutual Information Collaboration


<p align="center">
  <br><img src='fig/PMIC.png' width="600"/><br>
  % <a href="https://arxiv.org/abs/2203.04955">[Paper]</a>&emsp;<a href="https://nicklashansen.github.io/td-mpc">[Website]</a>
</p>

## Method

**PMIC** is a MARL framework For more effective MI-driven collaboration.
In PMIC, we use a new collaboration criterion measured by the MI between global states and joint actions.
Based on the criterion, the key idea of PMIC is maximizing the MI associated with superior collaborative behaviors and minimizing the MI associated with inferior ones.
The two MI objectives play complementary roles
by facilitating learning towards better collaborations while avoiding falling into sub-optimal ones.
Specifically, PMIC stores and progressively maintains sets of superior and inferior interaction experiences, from which dual MI neural estimators are established.
