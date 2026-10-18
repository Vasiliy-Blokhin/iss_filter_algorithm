""" «Разум» реверсивного алгоритма: ночной разбор опыта локальной моделью (AI_Manager).

Раз в сутки модель с рассуждением (REVERSE_MIND_MODEL, по умолчанию gpt-oss-20b) получает:
опыт алгоритма (живые, кандидаты и недавно умершие правила с результатами вне выборки, свои
сигналы и их исходы), что реально выросло и упало за последние дни и с какими признаками и
новостями, важные новости, словарь признаков с текущими квантилями и параметры.
Она возвращает:
    narrative  — что происходит на рынке и почему работали / не работали правила;
    hypotheses — до REFLECT_HYPOTHESES новых правил из 1-3 условий словаря: они рождаются
                 кандидатами (origin = llm) и торгуют только после той же проверки вне выборки,
                 что и найденные статистикой — модель не может «протащить» сделку;
    params     — предложения по своему пулу параметров (только значения из REVERSE_GRID):
                 принимаются, если слепой прогон за последние ~2 года не хуже действующих
                 параметров на SELFTUNE_MARGIN и ни одна половина не хуже больше чем на допуск.
"""
import json
import logging
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd

from elements.ai.worker import AIReviewer
from elements.reverse.engine import (
    ATOM_FEATURES, NEWS_FEATURES, FEATURE_TITLES, REVERSE_GRID, DEFAULT_REVERSE,
    Rule, RMarket, ReverseEngine, simulate_entries, describe,
)
from elements.reverse.news import MARKET
from elements.reverse.worker import ReverseAlgorithm, BOOTSTRAP_DAYS
from elements.selftune.blind import score_of
from elements.selftune.params import load_active, save_active
from source.settings.settings import (
    handler, REVERSE_MIND_MODEL, SELFTUNE_MARGIN, SELFTUNE_HALF_TOLERANCE,
)
from source.sql.main import SQLmain
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

REFLECT_HYPOTHESES = 3
CHECK_DAYS = 500  # период слепой проверки предложенных параметров

SYSTEM_PROMPT = (
    'Ты — «разум» самообучающегося алгоритма краткосрочных покупок акций на Московской бирже '
    '(вход на открытии следующего дня, удержание несколько дней, стоп/тейк по ATR, издержки '
    '~0.2% на сделку). Алгоритм работает «от обратного»: смотрит, какие сделки реально '
    'заработали, и ищет условия, которые им предшествовали; правило торгует только после '
    'подтверждения на новых данных. Тебе дан его опыт. Разбери его как исследователь: что '
    'происходит на рынке, какие закономерности держатся, какие исчезли и почему, как влияет '
    'новостной фон. Предложи новые проверяемые гипотезы — только из условий словаря признаков '
    '(признак, "<" или ">", порог; 1-3 условия) — и, если обосновано, изменения параметров '
    '(только допустимые значения). Не повторяй уже существующие правила. Помни: средняя сделка '
    'по рынку после издержек отрицательна, поэтому ценны условия с явной причиной '
    '(новости, перепроданность после паники, поток покупок), а не подгонка. '
    'Ответ — строго JSON: {"narrative": "3-6 предложений по-русски", '
    '"hypotheses": [{"atoms": [["rsi2", "<", 10], ["news_score", ">", 0.2]], "why": "кратко"}], '
    '"params": {"имя": значение}, "watch": "на что смотреть завтра"}'
)


def feature_dictionary(market: RMarket, day: int) -> str:
    """ Признаки с описанием и квантилями 10/50/90% за последний год (только прошлое)."""
    rows = np.nonzero(market.live_ok & (market.didx > day - 250) & (market.didx <= day))[0]
    lines = []
    for name in ATOM_FEATURES + NEWS_FEATURES:
        values = market.values.get(name)
        if values is None:
            continue
        values = values[rows]
        values = values[np.isfinite(values)]
        quantiles = (
            ', '.join(f'{q:.4g}' for q in np.quantile(values, [0.1, 0.5, 0.9])) if len(values) > 50
            else 'данных пока мало'
        )
        lines.append(f'{name} — {FEATURE_TITLES.get(name, name)} (10/50/90%: {quantiles})')
    return '\n'.join(lines)


def rule_lines(rules: list[Rule], limit: int) -> list[str]:
    return [
        f'{describe(rule.atoms)} [{rule.origin}, {rule.status}]: в окне при рождении {rule.is_n} сд. '
        f'средняя {rule.is_mean:+.2%}; вне выборки {rule.oos_n} дн. средняя {rule.oos_mean:+.2%}'
        + (f'; {rule.note}' if rule.note else '')
        for rule in rules[:limit]
    ]


def movers(market: RMarket, day: int, days: int = 5, top: int = 6) -> list[str]:
    """ Что реально выросло / упало за последние дни: признаки на день до движения и новости."""
    m = market
    rows = np.nonzero(m.known & (m.didx >= day - days - 3) & (m.eidx <= day))[0]
    if not len(rows):
        return []
    order = rows[np.argsort(m.ret[rows])]
    lines = []
    for title, chosen in (('лучшие', order[::-1][:top]), ('худшие', order[:top])):
        for row in chosen:
            features = ', '.join(
                f'{name}={m.values[name][row]:.3g}' for name in
                ('r1', 'r5', 'rsi2', 'vol_ratio', 'clv', 'dist_sma50', 'dist_high20', 'news_score')
                if name in m.values and np.isfinite(m.values[name][row])
            )
            lines.append(f'{title}: {m.panel["SECID"].iat[row]} сигнал {m.date(m.didx[row])}, '
                         f'сделка {m.ret[row]:+.2%}; {features}')
    return lines


def important_news(limit: int = 25) -> list[str]:
    rows = SQLmain.run_sql_command(
        'SELECT n.PUBLISHED, n.TITLE, s.SECID, s.SENTIMENT, s.IMPORTANCE FROM news_scores s '
        'JOIN news n ON n.id = s.NEWS_ID WHERE n.PUBLISHED >= datetime(\'now\', \'-3 day\') '
        'ORDER BY s.IMPORTANCE DESC, n.PUBLISHED DESC LIMIT :limit', {'limit': limit})
    return [f'{row["SECID"]} {row["SENTIMENT"]:+.1f} (важн. {row["IMPORTANCE"]:.1f}): {row["TITLE"]}'
            for row in rows]


def build_prompt(engine: ReverseEngine, day: int) -> str:
    m = engine.market
    live = sorted([r for r in engine.rules if r.status == 'live'], key=lambda r: -r.post)
    candidates = sorted([r for r in engine.rules if r.status == 'candidate'], key=lambda r: -r.is_post)
    dead = sorted([r for r in engine.rules if r.status == 'dead' and r.died is not None and r.died > day - 63],
                  key=lambda r: -r.died)
    own = [(d, row) for d, row, _, _ in engine.picks if d > day - 63]
    own_lines = [
        f'{m.date(d)} {m.panel["SECID"].iat[row]}: '
        + (f'{m.ret[row]:+.2%}' if np.isfinite(m.ret[row]) and m.eidx[row] <= day else 'ещё открыта')
        for d, row in own[-25:]
    ]
    count, mean = engine.own_record(day)
    params = {key: engine.p[key] for key in REVERSE_GRID}
    grid = {key: values for key, values in REVERSE_GRID.items()}
    sections = [
        f'День решения: {m.date(day)}. Режим: {"реальные сделки" if engine.trading else "теневые сделки"}; '
        f'свои сигналы за {engine.p["adapt_window"]} дн.: {count}, средняя {mean:+.2%}; '
        f'строгость {engine.level}; вес новостей {engine.news_weight:+.4f}.',
        'Словарь признаков:\n' + feature_dictionary(m, day),
        'Живые правила:\n' + ('\n'.join(rule_lines(live, 15)) or 'нет'),
        'Лучшие кандидаты:\n' + ('\n'.join(rule_lines(candidates, 10)) or 'нет'),
        'Недавно умершие:\n' + ('\n'.join(rule_lines(dead, 10)) or 'нет'),
        'Свои последние сигналы:\n' + ('\n'.join(own_lines) or 'нет'),
        'Реальные движения:\n' + ('\n'.join(movers(m, day)) or 'нет данных'),
        'Важные новости за 3 дня:\n' + ('\n'.join(important_news()) or 'новостей пока нет'),
        f'Параметры: {json.dumps(params, ensure_ascii=False)}\n'
        f'Допустимые значения: {json.dumps(grid, ensure_ascii=False)}',
    ]
    return '\n\n'.join(sections)


def parse_reply(content: str) -> dict:
    match = re.search(r'\{.*\}', content, re.S)
    return json.loads(match.group(0) if match else content)


def valid_atoms(atoms) -> list | None:
    if not isinstance(atoms, list) or not 1 <= len(atoms) <= 3:
        return None
    result = []
    for atom in atoms:
        if not isinstance(atom, (list, tuple)) or len(atom) != 3:
            return None
        feature, op, value = atom
        if feature not in ATOM_FEATURES + NEWS_FEATURES or op not in ('<', '>'):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        result.append([feature, op, value])
    if len({(a[0], a[1]) for a in result}) != len(result):
        return None
    return result


def snap_params(proposal: dict, current: dict) -> dict:
    """ Только известные параметры, значения — ближайшие из сетки."""
    result = {}
    for key, value in (proposal or {}).items():
        if key not in REVERSE_GRID:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        grid = REVERSE_GRID[key]
        nearest = min(grid, key=lambda option: abs(option - value))
        if nearest != current.get(key):
            result[key] = nearest
    return result


def reverse_blind(market: RMarket, params: dict) -> dict:
    """ Слепой прогон с нуля за последние CHECK_DAYS дней: весь период и две половины
    (в том же виде, что BlindBacktest.run — для общей проверки в самокоррекции)."""
    last = len(market.dates) - 1
    start = max(0, last - CHECK_DAYS)
    middle = (start + last) // 2
    params = {**DEFAULT_REVERSE, **params}
    first = max(params['lookback'] + 5, start - BOOTSTRAP_DAYS)
    engine = ReverseEngine(market, params).run(first, last)
    parts = [
        simulate_entries(market, engine.picks, start, last)['stats'],
        simulate_entries(market, engine.picks, start, middle)['stats'],
        simulate_entries(market, engine.picks, middle + 1, last)['stats'],
    ]
    parts = [{**stats, 'SCORE': score_of(stats)} for stats in parts]
    return {'stats': parts[0], 'halves': parts[1:],
            'blind_start': market.date(start), 'data_end': market.date(last)}


def check_params(panel: pd.DataFrame, current: dict, candidate: dict) -> tuple[bool, str]:
    """ Слепой прогон за последние CHECK_DAYS дней: кандидат против действующих параметров."""
    market = ReverseAlgorithm.market(panel)
    results = {}
    for name, params in (('current', current), ('candidate', candidate)):
        result = reverse_blind(market, params)
        results[name] = [result['stats'], *result['halves']]
    current_score = [stats['SCORE'] for stats in results['current']]
    candidate_score = [stats['SCORE'] for stats in results['candidate']]
    accepted = (
        candidate_score[0] >= current_score[0] + SELFTUNE_MARGIN
        and all(c >= b - SELFTUNE_HALF_TOLERANCE for c, b in zip(candidate_score[1:], current_score[1:]))
    )
    text = (f'слепо за {CHECK_DAYS} дн.: CAGR {results["candidate"][0]["CAGR"]:+.1%} против '
            f'{results["current"][0]["CAGR"]:+.1%}, просадка {results["candidate"][0]["MAX_DRAWDOWN"]:.1%} '
            f'против {results["current"][0]["MAX_DRAWDOWN"]:.1%}')
    return accepted, text


def reflect(panel: pd.DataFrame, model: str = REVERSE_MIND_MODEL, rng: random.Random | None = None) -> dict:
    """ Ночной разбор: нарратив, гипотезы -> кандидаты, предложения параметров -> слепая проверка."""
    rng = rng or random.Random()
    market = ReverseAlgorithm.market(panel)
    engine, stored = ReverseAlgorithm.load_engine(market)
    day = ReverseAlgorithm.day_of(market, stored.get('last_day'))
    if day is None:
        return {'error': 'алгоритм ещё не запускался'}
    journal_from, picks_from = len(engine.journal), len(engine.picks)
    prompt = build_prompt(engine, day)
    reviewer = AIReviewer(model=model)
    content = reviewer.chat(
        [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}],
        max_tokens=3000, extra={'temperature': 0.4},
    )
    reply = parse_reply(content)
    narrative = str(reply.get('narrative') or '')[:1500]
    watch = str(reply.get('watch') or '')[:500]
    engine.log(day, 'reflection', f'{model}: {narrative}' + (f' Смотреть: {watch}' if watch else ''))

    born = []
    for hypothesis in (reply.get('hypotheses') or [])[:REFLECT_HYPOTHESES]:
        atoms = valid_atoms(hypothesis.get('atoms') if isinstance(hypothesis, dict) else None)
        if not atoms:
            continue
        mask = market.mask(atoms) & market.known & (market.eidx <= day) & (market.didx > day - engine.p['lookback'])
        returns = market.ret[mask]
        rule = Rule(atoms, born=day, origin='llm', is_n=int(mask.sum()),
                    is_mean=float(returns.mean()) if len(returns) else 0.0,
                    note=str(hypothesis.get('why') or '')[:300])
        rule.is_post = rule.is_mean
        if engine.add_rule(rule):
            born.append(describe(atoms))

    # Предложения параметров (модели или, если их нет, случайная мутация) — слепая проверка.
    current = load_active('reverse')
    proposal = snap_params(reply.get('params'), current)
    source = model
    if not proposal:
        key = rng.choice(list(REVERSE_GRID))
        proposal = {key: rng.choice([v for v in REVERSE_GRID[key] if v != current.get(key)])}
        source = 'мутация'
    candidate = {**current, **proposal}
    accepted, text = check_params(panel, current, candidate)
    if accepted:
        save_active('reverse', candidate, data_end=market.date(day), note=f'{source}: {proposal}; {text}')
    engine.log(day, 'params', f'{source} предлагает {proposal}: {text} — '
               + ('принято' if accepted else 'отклонено'))
    ReverseAlgorithm.save_engine(engine, day, journal_from, picks_from)
    return {'narrative': narrative, 'watch': watch, 'hypotheses': born, 'params': proposal,
            'accepted': accepted, 'check': text}
