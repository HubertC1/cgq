from agents.acfql import ACFQLAgent
from agents.dqc import DQCAgent
from agents.cgq import CGQAgent
from agents.iql import IQLAgent
from agents.aciql import ACIQLAgent
from agents.sarsa import SARSAAgent
from agents.bc import BCAgent
agents = dict(
    acfql=ACFQLAgent,
    dqc=DQCAgent,
    cgq=CGQAgent,
    iql=IQLAgent,
    aciql=ACIQLAgent,
    sarsa=SARSAAgent,
    bc=BCAgent,
)