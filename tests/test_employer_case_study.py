def test_employer_case_study_is_linked_and_uses_portfolio_framing():
    case_study = open("CASE_STUDY.md", encoding="utf-8").read()
    readme = open("README.md", encoding="utf-8").read()
    start = open("START_HERE.md", encoding="utf-8").read()

    assert "# Case study · Rule-conditioned moderation NLP" in case_study
    assert "0.91808 public / 0.91425 private ROC AUC" in case_study
    assert "AWS SageMaker" in case_study
    assert "negative results" in case_study.lower()
    assert "CASE_STUDY.md" in readme
    assert "CASE_STUDY.md" in start
    assert "Measured improvement" in case_study
    assert "leaderboard-equivalent evidence" in case_study
