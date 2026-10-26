"""EXTENSION 1 - Laya decides whether the retrieved evidence is enough to answer.

Laya (https://github.com/NandhaKishorM/laya, `pip install laya`) reads a text 'state' plus typed questions and returns
a typed answer in one forward pass. Here: ONE 'choice' question with two labels, 'answer' / 'abstain'.
Rules: if Laya fails for any reason -> abstain and record the error (never counted as a correct abstention)."""
import config

QUESTIONS = {"enough_evidence": {
    "type": "choice",
    "instructions": "Does the evidence text state the specific fact the question asks for? "
                    "Topic overlap alone is not enough.",
    "criteria": {"answer": "the evidence states the specific fact the question asks for",
                 "abstain": "the evidence does not state the specific fact asked for"}}}


def build_state(question, retrieved):
    parts = [f"Question: {question}"]
    parts += [f"Evidence {r['rank']} (PDF page {r['chunk']['page']}): {r['chunk']['text']}" for r in retrieved]
    return "\n\n".join(parts)


class LayaGate:
    def __init__(self, agent=None):
        if agent is None:                                        # real model: needs torch + Hugging Face access
            import laya
            agent = laya.load(config.LAYA_REPO, subfolder=config.LAYA_SUBFOLDER)
        self.agent = agent

    def decide(self, question, retrieved):
        """-> {"label": "answer"|"abstain", "error": None|str, "input": str, "output": raw Laya result}"""
        state = build_state(question, retrieved)
        try:
            output = self.agent.predict(state, QUESTIONS)
            label = output["answers"]["enough_evidence"]["choice"]
            if label not in ("answer", "abstain"):
                raise ValueError(f"unexpected label {label!r}")
        except Exception as exc:                                 # any failure -> abstain + log
            return {"label": "abstain", "error": repr(exc), "input": state, "output": None}
        return {"label": label, "error": None, "input": state, "output": output}
