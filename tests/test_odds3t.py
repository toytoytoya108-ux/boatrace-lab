"""3連単オッズ表のパース。実ページの構造（class無しのtable・見出しに選手名・2着はrowspan）で検証する。

2026-09-10: 本番で120通りとも読めておらず、絞り込み型がずっと推定オッズで動いていた。
原因は表を class="is-w495" で探していたこと（実ページの table に class が無い）と、
見出し行を「艇番6個だけの行」と決め打ちしていたこと（実際は選手名が交互に入る）。
"""
import pytest

from boatlab.ingest.official_web import parse_odds3t

NAMES = ["上田　龍星", "中村　泰平", "船岡　洋一郎", "北野　輝季", "桑原　　徹", "森　　健二"]


def _expected():
    return {f"{a}-{b}-{c}": round(5 + a * 10 + b + c / 10, 1)
            for a in range(1, 7) for b in range(1, 7) if b != a
            for c in range(1, 7) if c not in (a, b)}


def _page(head_with_names=True, table_class="", nav=True):
    exp = _expected()
    head = "<tr>" + "".join((f"<th>{i + 1}</th><th>{NAMES[i]}</th>" if head_with_names else f"<th>{i + 1}</th>")
                            for i in range(6)) + "</tr>"
    cols = [[(b, c, exp[f"{a}-{b}-{c}"]) for b in range(1, 7) if b != a for c in range(1, 7) if c not in (a, b)]
            for a in range(1, 7)]
    rows = []
    for i in range(20):
        cells = ""
        for k in range(6):
            b, c, o = cols[k][i]
            if i % 4 == 0:
                cells += f'<td rowspan="4">{b}</td>'          # 2着セルは4行に1回だけ
            cells += f"<td>{c}</td><td>{o}</td>"
        rows.append(f"<tr>{cells}</tr>")
    navtbl = ("<table><tr><td>レース</td>" + "".join(f"<td>{i + 1}R</td>" for i in range(12))
              + "</tr><tr><td>締切予定時刻</td>" + "".join("<td>15:15</td>" for _ in range(12)) + "</tr></table>") if nav else ""
    cls = f' class="{table_class}"' if table_class else ""
    return f"<html>{navtbl}<table{cls}>{head}{''.join(rows)}</table></html>", exp


def test_parse_real_page_shape():
    html, exp = _page()
    assert parse_odds3t(html) == exp


@pytest.mark.parametrize("kwargs", [
    {"table_class": "is-w495"},        # 旧構造（class あり）も引き続き読めること
    {"head_with_names": False},        # 見出しが艇番だけの場合
    {"nav": False},                    # レース番号の表が無い場合
])
def test_parse_tolerates_variations(kwargs):
    html, exp = _page(**kwargs)
    assert parse_odds3t(html) == exp


def test_missing_boat_is_none_not_dropped():
    """欠場でオッズが空欄でも、組番は残して値を None にする（推測で埋めない）。"""
    html, exp = _page()
    html = html.replace(f"<td>{exp['1-2-3']}</td>", "<td>－</td>", 1)
    got = parse_odds3t(html)
    assert len(got) == 120 and got["1-2-3"] is None


def test_unrelated_page_returns_empty():
    assert parse_odds3t("<html><table><tr><td>まったく別の表</td></tr></table></html>") == {}
