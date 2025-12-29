from forecastbench.models.ar_cot import _extract_prob


def test_extract_prob_json():
    assert _extract_prob('{"p_yes": 0.73}') == 0.73
    assert _extract_prob("{'p_yes': 0.2}") == 0.2
    assert _extract_prob("p_yes: 0.01") == 0.01
    assert _extract_prob('some text {"p_yes": 70%}') == 0.7


def test_extract_prob_tail_number():
    txt = "OUTPUT:\\n... reasoning ...\\n0.42\\n"
    assert _extract_prob(txt) == 0.42


def test_extract_prob_percent_phrase():
    assert _extract_prob("Probability of YES: 84%") == 0.84
    assert _extract_prob("P(YES)=0.9") == 0.9



