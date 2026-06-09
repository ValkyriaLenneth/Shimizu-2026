from rfdetr_prod_pipeline.pipeline.wall_candidate_display import wall_display_grade


def test_wall_display_grade_matrix() -> None:
    expected = {
        ("B", "B"): "B",
        ("B", "C"): "C",
        ("B", "D"): "D",
        ("C", "B"): "B",
        ("C", "C"): "C",
        ("C", "D"): "D",
        ("D", "B"): "D",
        ("D", "C"): "D",
        ("D", "D"): "D",
    }
    for pair, grade in expected.items():
        assert wall_display_grade(*pair) == grade
