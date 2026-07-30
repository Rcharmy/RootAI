"""One-off: verify get_structured_llm returns a Pydantic instance, not a string."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel
from rootai.tools.llm import get_structured_llm


class Answer(BaseModel):
    kpi: str
    is_financial: bool


llm = get_structured_llm(Answer)
resp = llm.invoke("Is revenue a KPI, and is it financial?")
print("type:", type(resp).__name__)
print("kpi:", resp.kpi)
print("is_financial:", resp.is_financial)