from agents.acfql import ACFQLAgent
from agents.dqc import DQCAgent
from agents.dqc_nodistill import DQCNoDistillAgent
from agents.dqc_gripcond import DQCGripCondAgent
from agents.cgq import CGQAgent
from agents.cgq_gripcond import CGQGripCondAgent
from agents.iql import IQLAgent
from agents.aciql import ACIQLAgent
from agents.sarsa import SARSAAgent
from agents.bc import BCAgent
agents = dict(
    acfql=ACFQLAgent,
    dqc=DQCAgent,
    dqc_nodistill=DQCNoDistillAgent,
    dqc_gripcond=DQCGripCondAgent,
    cgq=CGQAgent,
    cgq_gripcond=CGQGripCondAgent,
    iql=IQLAgent,
    aciql=ACIQLAgent,
    sarsa=SARSAAgent,
    bc=BCAgent,
)