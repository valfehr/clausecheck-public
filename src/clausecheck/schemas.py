"""Domain model for ClauseCheck.

These schemas are the contract between every component (parser, retriever, judge,
API, UI, evals). They are deliberately independent of any LLM provider or framework.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    GREEN = "green"    # same legal effect as the golden rule
    YELLOW = "yellow"  # same intent, but weaker/ambiguous wording or carve-outs -> human review
    RED = "red"        # missing, or legal effect contradicts the rule


class RedReason(str, Enum):
    MISSING = "missing"
    CONTRADICTS = "contradicts"


class Rubric(BaseModel):
    green: str
    yellow: str
    red: str


class Rule(BaseModel):
    """A 'golden rule' provided by the client's legal team."""

    id: str
    title: str
    ideal_text: str = Field(description="Example of ideal clause wording")
    rubric: Rubric = Field(description="What distinguishes green / yellow / red for this rule")
    doc_types: list[str] = Field(default_factory=list, description="Doc types where this rule is preselected")
    cuad_category: str | None = Field(default=None, description="Mapping to CUAD label, for evals")


class Clause(BaseModel):
    """A segment of a contract. Excerpt boundaries come from here, never from the LLM."""

    id: str                      # e.g. "doc42:12.3(b)"
    doc_id: str
    path: list[str]              # hierarchy, e.g. ["Article 12", "12.3", "(b)"]
    text: str
    start_char: int
    end_char: int


class Verdict(BaseModel):
    """Structured output of the judge for one (document, rule) pair."""

    rule_id: str
    status: Status
    red_reason: RedReason | None = None
    cited_clause_ids: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class EvalCase(BaseModel):
    """One labelled example for the eval harness."""

    case_id: str
    suite: str = Field(description="'severity' | 'retrieval' (later: 'chat', 'redraft', 'security')")
    split: str = Field(description="'dev' | 'test' | 'holdout', assigned by contract family")
    doc_id: str
    family: str
    rule_id: str
    client_party: str = Field(description="Which party the reviewing lawyer represents")
    text: str = Field(description="Full contract text (possibly with a synthetic clause swapped in)")
    expected_status: Status | None = Field(default=None, description="None = not scored for severity")
    expected_red_reason: RedReason | None = None
    expected_spans: list[tuple[int, int]] = Field(default_factory=list)
    source: str = Field(description="'cuad' | 'synthetic' | 'human'")
    note: str = ""
