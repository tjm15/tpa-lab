from tpa.material_considerations import MaterialConsideration, Finding
from tpa.balance import build_planning_balance


def test_build_planning_balance_fallback():
    mc = MaterialConsideration(
        topic="transport",
        weight="significant",
        recommendation="unacceptable",
        findings=[
            Finding(
                claim="Applicant relies on junction improvements",
                supporting=["APP:TRANSPORT_P1"],
                contradicting=["POL:T1_P12"],
                conclusion="Harm remains due to unresolved safety evidence.",
                weight="significant",
            )
        ],
    )
    markdown, summary, narrative = build_planning_balance([mc], llm=None)
    assert "transport" in markdown.lower()
    assert summary["harms"]
    assert "On balance" in narrative
