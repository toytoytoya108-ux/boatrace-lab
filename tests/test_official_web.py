"""公式 3連単オッズ表の想定構造に対するパーサ検証（実ページは環境の都合で未検証）。"""
import itertools

from boatlab.ingest.official_web import parse_odds3t


def _synthetic_html():
    # 1着 1..6 の列、2着ブロック（rowspan=4）、3着行、オッズ
    rows = ["<tr>" + "".join(f"<th>{b}</th>" for b in range(1, 7)) + "</tr>"]
    odds = {}
    for a in range(1, 7):
        for b in range(1, 7):
            if b == a:
                continue
            for c in range(1, 7):
                if c in (a, b):
                    continue
                odds[f"{a}-{b}-{c}"] = round(5 + (a * 36 + b * 6 + c) * 0.7, 1)
    # 行構造：2着候補の順序は列ごとに異なる（1着艇を除く）ので、行 r（0..4）と行内 j（0..3）で組む
    for r in range(5):
        for j in range(4):
            tds = []
            for a in range(1, 7):
                seconds = [b for b in range(1, 7) if b != a]
                b = seconds[r]
                thirds = [c for c in range(1, 7) if c not in (a, b)]
                c = thirds[j]
                if j == 0:
                    tds.append(f'<td rowspan="4">{b}</td>')
                tds.append(f"<td>{c}</td><td class=\"oddsPoint\">{odds[f'{a}-{b}-{c}']}</td>")
            rows.append("<tr>" + "".join(tds) + "</tr>")
    return '<table class="is-w495">' + "".join(rows) + "</table>", odds


def test_parse_synthetic_structure():
    html, expected = _synthetic_html()
    got = parse_odds3t(html)
    assert len(got) == 120
    assert got == expected


def test_parse_empty_returns_empty():
    assert parse_odds3t("<html></html>") == {}
