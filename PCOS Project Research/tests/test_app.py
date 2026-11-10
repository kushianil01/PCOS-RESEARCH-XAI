"""UI smoke tests with Streamlit's headless AppTest (no browser)."""
import pytest
from streamlit.testing.v1 import AppTest
from pcos import config as C

APP = str(C.ROOT / "app/streamlit_app.py")
PAGES = ["Overview", "Dataset & data quality", "Model evaluation", "Prediction", "Explainability", "Responsible use"]


def _run(page):
    at = AppTest.from_file(APP, default_timeout=60); at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page):
    at = _run(page)
    assert not at.exception, at.exception


def test_prediction_page_blocks_empty_form_and_lists_missing_fields():
    at = _run("Prediction")
    at.button[0].click().run()
    assert not at.exception
    assert any("correct" in e.value.lower() for e in at.error)
    assert any("missing" in m.value.lower() for m in at.markdown)


def test_prediction_page_valid_submission_shows_score_and_threshold():
    at = _run("Prediction")
    at.number_input[0].set_value(28); at.number_input[1].set_value(158.0); at.number_input[2].set_value(60.0)
    for sb in at.selectbox:
        sb.select("Yes" if "Yes" in sb.options else "Irregular")
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert any("probability" in l.lower() for l in labels) and any("threshold" in l.lower() for l in labels)


def test_prediction_page_rejects_out_of_range_value():
    at = _run("Prediction")
    at.number_input[0].set_value(5); at.number_input[1].set_value(158.0); at.number_input[2].set_value(60.0)
    for sb in at.selectbox: sb.select("No" if "No" in sb.options else "Regular")
    at.button[0].click().run()
    assert any("outside the allowed range" in m.value for m in at.markdown)


def test_no_lab_or_ultrasound_input_widgets_exist():
    at = _run("Prediction")
    labels = " ".join([w.label for w in list(at.number_input) + list(at.selectbox)]).lower()
    for banned in ["amh", "lh", "fsh", "follicle", "hormone", "pregnan", "blood", "pulse"]:
        assert banned not in labels
