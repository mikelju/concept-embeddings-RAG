"""Phase 17, S6 (R5): tables, macros and the guard, on a small fixture."""

from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import paper

SYSTEMS = {
    "dense": "P10-A",
    "hybrid-bm25": "P10-B",
    "hybrid-bm25-entity-hop": "P10-C",
    "hybrid-bm25-seeded-hop": "P14",
}
N = 7405
KS = [2, 5, 10, 20, 100]


def comparison(candidate: str, control: str, wins: int, losses: int, p: float) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "control": control,
        "wins": wins,
        "losses": losses,
        "ties": N - wins - losses,
        "delta_percentage_points": 100 * (wins - losses) / N,
        "exact_two_sided_p": p,
        "n_questions": N,
    }


def line(supported: int, label: str, descriptive: bool = False) -> dict[str, Any]:
    return {
        "label": label,
        "descriptive": descriptive,
        "budgets": {
            str(b): {"full_support_count": supported, "full_support": supported / N}
            for b in config.PHASE_17_BUDGETS
        },
        "ceiling": {
            "all_gold_inside": supported + 100,
            "failures_inside": 100,
            "full_support_2048": supported,
            "n_questions": N,
        },
        "full_support_at_k": {str(k): 0.5 for k in KS},
        "gold_recall_at_k": {str(k): 0.75 for k in KS},
        "ndcg_at_10": 0.8316,
    }


def fixture() -> paper.Sources:
    supported = {"dense": 4125, "hybrid-bm25": 4536, "hybrid-bm25-entity-hop": 4801}
    supported["hybrid-bm25-seeded-hop"] = 5224
    fs: dict[str, Any] = {}
    lines: dict[str, Any] = {}
    for system, count in supported.items():
        fs[system] = {"supported": count, "descriptive": False, "n_questions": N}
        lines[system] = line(count, SYSTEMS[system])
        fs[f"light:{system}"] = {"supported": count + 10, "descriptive": False}
        lines[f"light:{system}"] = line(count + 10, f"J-light({SYSTEMS[system]})")
    fs["light:union"] = {"supported": 5300, "descriptive": True}
    lines["light:union"] = line(5300, "J-light(union)", True)
    per_judge = {
        "light": {
            "j_p10a_vs_p10a": comparison("J-light(P10-A)", "P10-A", 400, 10, 2.97e-50),
            "j_p10b_vs_p10b": comparison("J-light(P10-B)", "P10-B", 200, 6, 0.0),
            "j_p10c_vs_p10c": comparison("J-light(P10-C)", "P10-C", 100, 60, 0.0242),
            "j_p14_vs_j_p10b": comparison("J-light(P14)", "J-light(P10-B)", 383, 59, 1.2e-3),
            "j_p14_vs_p14": comparison("J-light(P14)", "P14", 30, 40, 0.5),
        }
    }
    outcome = {
        "judges": {"light": {"label": "J-light"}},
        "systems": SYSTEMS,
        "sets": {
            "hotpotqa": {
                "label": "HotpotQA dev",
                "n_questions": N,
                "full_support_2048": fs,
                "per_judge": per_judge,
                "hop_label": {"light": "HOP_ADDS_UNDER_JUDGE"},
                "cross_judge": {
                    "strong_vs_light_p10b": comparison(
                        "J-strong(P10-B)", "J-light(P10-B)", 9, 1, 0.01
                    )
                },
            }
        },
    }
    metrics = {
        "ks": KS,
        "budgets": list(config.PHASE_17_BUDGETS),
        "sets": {"hotpotqa": {"lines": lines}},
    }
    return paper.Sources(
        {
            "outcome": paper.Source("data/phase17/outcome.json", "ab" * 32, outcome),
            "metrics": paper.Source("data/phase17/metrics.json", "cd" * 32, metrics),
        }
    )


def test_main_table_holds_the_fixture_values_at_the_declared_rounding() -> None:
    sources = fixture()
    table = paper.table_main(sources["outcome"], sources["metrics"])
    markdown = paper.render_md_table(table)
    assert "4,125 (55.71)" in markdown
    assert "4,135 (55.84)" in markdown  # J-light(P10-A): 4,135 / 7,405
    assert "5,300 (71.57) (descriptive)" in markdown
    tex = paper.render_tex_table(table)
    assert "5,224 (70.55)" in tex
    assert "$^\\dagger$" in tex
    assert tex.startswith("% Generated") and "\\toprule" in tex and "\\bottomrule" in tex


def test_d5_table_labels_the_hop_row_and_formats_p_values() -> None:
    sources = fixture()
    markdown = paper.render_md_table(paper.table_d5(sources["outcome"], "hotpotqa"))
    hop = next(row for row in markdown.splitlines() if row.startswith("| J-light(P14) vs J-light"))
    assert "HOP_ADDS_UNDER_JUDGE" in hop and "+4.38" in hop and "1.2e-3" in hop
    assert "< 1e-300" in markdown
    tex = paper.render_tex_table(paper.table_d5(sources["outcome"], "hotpotqa"))
    assert "$<10^{-300}$" in tex
    assert "HOP\\_ADDS\\_UNDER\\_JUDGE" in tex


def test_d6_table_has_one_row_per_line_with_the_percentages() -> None:
    sources = fixture()
    markdown = paper.render_md_table(
        paper.table_d6(sources["outcome"], sources["metrics"], "hotpotqa")
    )
    assert "75.00" in markdown and "50.00" in markdown and "83.16" in markdown
    assert markdown.count("| P1") == 4


def test_p_value_formatting() -> None:
    assert paper.p_value(0.0).tex == "$<10^{-300}$"
    assert paper.p_value(0.0).md == "< 1e-300"
    assert paper.p_value(2.97e-50).tex == "$3.0\\times10^{-50}$"
    assert paper.p_value(2.97e-50).md == "3.0e-50"
    assert paper.p_value(0.0242).tex == "$2.4\\times10^{-2}$"
    assert paper.p_value(9.96e-4).md == "1.0e-3"


def test_delta_and_counts() -> None:
    assert paper.delta_pp(9.62862930452397).tex == "+9.63"
    assert paper.delta_pp(-3.2815964523281598).tex == "$-$3.28"
    assert paper.delta_pp(-0.001).tex == "0.00"
    assert paper.count(1295224).tex == "1,295,224"
    assert paper.share(5198, 7405, unit=True).tex == "70.20\\%"
    assert paper.words("d5-hotpotqa") == "DFiveHotpotqa"


def test_every_macro_equals_its_source_key() -> None:
    sources = fixture()
    macros = paper.build_macros(sources)
    by_name = {m.name: m for m in macros}
    assert len(by_name) == len(macros)
    direct = [m for m in macros if m.key is not None]
    assert len(direct) > 50
    for macro in direct:
        assert macro.source is not None and macro.fmt is not None and macro.key is not None
        raw = paper.lookup(macro.source.data, macro.key)
        assert raw == macro.raw
        assert macro.fmt(raw).tex == macro.text
    assert by_name["fsPTenAHotpot"].text == "4,125"
    assert by_name["fsPctPTenAHotpot"].text == "55.71\\%"
    assert by_name["fsLightUnionHotpot"].text == "5,300"
    assert by_name["winsLightHopHotpot"].text == "383"
    assert by_name["deltaLightPTenAHotpot"].text == "+5.27"
    assert by_name["pLightPTenBHotpot"].text == "$<10^{-300}$"
    assert by_name["labelLightHotpot"].text == "HOP\\_ADDS\\_UNDER\\_JUDGE"
    assert by_name["nQuestionsHotpot"].text == "7,405"


def test_rendered_macros_carry_path_digest_and_key() -> None:
    sources = fixture()
    text = paper.render_macros(paper.build_macros(sources), sources)
    assert "% data/phase17/outcome.json sha256:abababababab sets.hotpotqa.n_questions" in text
    assert "\\newcommand{\\nQuestionsHotpot}{7,405}" in text
    assert "% derived:" in text


def test_the_output_is_deterministic(tmp_path: Path) -> None:
    sources = fixture()
    first = paper.render_macros(paper.build_macros(sources), sources)
    second = paper.render_macros(paper.build_macros(sources), sources)
    assert first == second
    figure = paper.figure_budget(sources["outcome"], sources["metrics"], "hotpotqa")
    assert figure.tex == paper.figure_budget(sources["outcome"], sources["metrics"], "hotpotqa").tex
    assert "\\addplot coordinates" in figure.tex and "(2048,61.26)" in figure.tex


def test_spec_ceilings_are_parsed_from_the_spec_table() -> None:
    spec = (
        "| Corpus | System | Full Support @2,048, measured | All gold |\n"
        "|---|---|---:|---:|\n"
        "| MultiHop-RAG, 2,255 | P10-B | 587 | 1,160 |\n"
        "| | union of the four top-100 | | 1,298 |\n"
        "| HotpotQA dev, 7,405 | P10-C | 4,801 | 932 failures with both gold |\n"
        "\nafter\n"
    )
    parsed = paper.parse_spec_ceilings(spec)
    assert parsed["multihop-rag"]["hybrid-bm25"] == (1160, "ceiling")
    assert parsed["multihop-rag"]["union"] == (1298, "ceiling")
    assert parsed["hotpotqa"]["hybrid-bm25-entity-hop"] == (932, "failures")


def test_r5_guard_passes_on_an_empty_directory(tmp_path: Path) -> None:
    assert paper.r5_violations(tmp_path) == []


def test_r5_guard_catches_hand_written_numbers(tmp_path: Path) -> None:
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "numbers.tex").write_text("\\newcommand{\\a}{5,198 70.20\\%}\n")
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "results.tex").write_text(
        "The hop gains 4.90 points, or 70\\% of questions, over 5,198 cases.\n"
    )
    (tmp_path / "main.tex").write_text(
        "Seven limitations. \\cite{thakur2021beir} 2021 % 4.5 note\n"
    )
    found = paper.r5_violations(tmp_path)
    assert len(found) == 3
    assert all(item.startswith("sections/results.tex:1") for item in found)


def test_r5_guard_allows_identifiers_options_and_years(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_text(
        "\\documentclass[10pt]{article}\n\\usepackage[margin=1.5cm]{geometry}\n"
        "\\pgfplotsset{compat=1.18}\n\\input{tables/d5-hotpotqa}\n"
        "As in~\\cite{xiong2021mdr}, see Table~\\ref{tab:d5} and \\label{fig:b2}. In 2024.\n"
        "The hop gains \\deltaStrongPTenBHotpot{} points, \\fsPctPFourteenHotpot{} of "
        "\\nQuestionsHotpot{} questions.\n"
    )
    assert paper.r5_violations(tmp_path) == []
